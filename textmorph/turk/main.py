"""
Initiate MTurk tasks from files containing sentence chains.
"""
import os
from os.path import basename, splitext
import re
import copy
import sys
import subprocess
import time
import math 
import traceback
import itertools
from collections import defaultdict

import simplejson as json
import numpy as np

from turk import SimilarityTask, CoherenceTask, mtc


class params:
    DATUM_PER_EXP = 5   # Number of pairs to show per HIT
    SENT_MAX_DIST = 5   # Bound on sentence pair distance
    TASK_SIMILARITY = 0
    TASK_COHERENCE = 1

    num_datum = 5       # Deploy this many pairs/sentences
    task_type = TASK_SIMILARITY
    debug = True
    database_file = "mturk_data.json"
    database = None
    data_file = None


def parse_args(args):
    help = """
        use put to deploy HITs from chains_file
        use get to update local data store
        use review to approve HITs and pay workers
        usage:  main.py put
                main.py get
                main.py review 

        --debug: [true/false] publish tasks for real or not
        --num-datum [#] of pairs to deploy
        --use-db [data file] to read / write results to database
        --task-type [similarity/coherence]
        --data-file [file] to read sentence data from 

        !Warning: "line numbers" are 1-indexed.
    """

    if 'help' in ''.join(args):
        print help 
        sys.exit(1)

    if len(args) > 0:
        cmd = args[0]
        inputs = args[1:]

        for i in range(0, len(inputs), 2):
            try:
                flg = inputs[i]
                arg = inputs[i+1]

                if flg == '--debug':
                    if arg == 'false':
                        params.debug = False 

                elif flg == '--task-type':
                    if arg == "similarity":
                        params.task_type = params.TASK_SIMILARITY
                    elif arg == "coherence":
                        params.task_type = params.TASK_COHERENCE

                elif flg == '--data-file':
                    params.data_file = arg 

                elif flg == '--num-datum':
                    params.num_datum = int(arg)

                elif flg == '--use-db':
                    params.database_file = arg 
            except:
                print help 
                sys.exit(1)

        return cmd

    print help 
    sys.exit(1)


""" Data store """


class MTurkData(object):

    def __init__(self, data_file_name):
        self.data = {
            'hits': {},
            'data': {}
        }
        self.data_file_name = data_file_name
        if not os.path.exists(data_file_name):
            self._make()
        with open(data_file_name, 'r') as data_f:
            self.data = json.loads(data_f.read())
        print "using database:", self.data_file_name

    def _make(self):
        print "creating database:", self.data_file_name
        with open(self.data_file_name, 'w') as data_f:
            data_f.write(json.dumps(self.data, sort_keys=True, indent=4))
            return

    def save(self):
        data_json = json.dumps(self.data, sort_keys=True, indent=4)
        with open(self.data_file_name + 'temp', 'w') as data_temp:
            data_temp.write(data_json)
        with open(self.data_file_name, 'w') as data_f:  # if failure during write, we have temp
            data_f.write(data_json)

    def add_hit(self, hit_id, is_complete=False):
        data = {
            'is_complete': is_complete
        }
        self.data['hits'][hit_id] = data

    def get_complete_hits(self):
        return [h for (h, k) in self.data['hits'].iteritems() if k['is_complete']]

    def get_incomplete_hits(self):
        return [h for (h, k) in self.data['hits'].iteritems() if not k['is_complete']]

    def complete_hit(self, hit_id):
        self.data['hits'][hit_id]['is_complete'] = True

    def get_without_workers(self, blacklist):
        data = copy.deepcopy(self.data)
        for p_id, p_data in data['pairs'].iteritems():
            for idx, rating in enumerate(p_data['ratings']):
                w_id = rating[1]
                if w_id in blacklist:
                    del data['pairs'][p_id]['ratings'][idx]
        return data

    def dedupe(self):
        data = copy.deepcopy(self.data)
        for p_id, p_data in self.data['pairs'].iteritems():
            add_workers = {}
            data['pairs'][p_id]['ratings'] = []
            for (rating, worker_id) in p_data['ratings']:
                if worker_id not in add_workers:
                    data['pairs'][p_id]['ratings'].append([rating, worker_id])
                    add_workers[worker_id] = True
        self.data = data

    # Similarity data handlers
    @classmethod
    def pair_id(self, ln_1, ln_2, chains_file):
        cf_id = splitext(basename(chains_file))[
            0]  # file the example came from
        ln_1, ln_2 = sorted([ln_1, ln_2])
        p_id = "{}_{}_{}".format(cf_id, ln_1, ln_2)
        return p_id

    def add_pair(self, pair_id, s_1, s_2):
        if pair_id in self.data['data']:
            return
        data = {
            'sentences': [s_1, s_2],
            'ratings': []
        }
        self.data['data'][pair_id] = data

    def add_rating(self, pair_id, rating, worker_id):
        r = (rating, worker_id)
        self.data['data'][pair_id]['ratings'].append(r)

    # Coherence data handlers
    @classmethod
    def sent_id(self, ln, sentences_file):
        sf_id = splitext(basename(sentences_file))[0]
        s_id = "{}_{}".format(sf_id, ln)
        return s_id 

    def add_sent(self, sent_id, s):
        if sent_id in self.data['data']:
            return 
        data = {
            'sentence': s,
            'coherence': [],
            'grammar': []
        }
        self.data['data'][sent_id] = data

    def add_sent_scores(self, sent_id, coherence_score, grammar_score, worker_id):
        c_sc = (coherence_score, worker_id)
        g_sc = (grammar_score, worker_id)
        self.data['data'][sent_id]['coherence'].append(c_sc)
        self.data['data'][sent_id]['grammar'].append(g_sc)

    def add_sent_sim_score(self, file_id, similarity_score, worker_id):
        sim_id = 'sim_' + file_id
        if 'similarities' not in self.data['data']:
            self.data['data']['similarities'] = {}
        if sim_id not in self.data['data']['similarities']:
            self.data['data']['similarities'][sim_id] = []
        self.data['data']['similarities'][sim_id].append((similarity_score, worker_id)) 


""" Util functions """

def assert_or_report(expr, msg):
    if not expr:
        print msg 
        sys.exit(1)


def user_confirmation(msg, yes="y"):
    response = raw_input(msg + ': ')
    return response == yes


def sentence_index_to_line_number(chain_idx, sent_idx, chains, line_numbers):
    abs_idx = sum([len(c) for c in chains[:chain_idx]]) + sent_idx
    return line_numbers[abs_idx]


def xml_safe(string):  # escape syntactic tags so they are not considered XML tags
    string = string.replace("&", "&amp;")
    string = string.replace("<", "&lt;")
    string = string.replace(">", "&gt;")
    string = string.replace("\"", "\\\"")
    # string = string.encode('ascii', 'xmlcharrefreplace')
    return string


def un_xml_safe(string):
    string = string.replace("&amp;", "&")
    string = string.replace("&lt;", "<")
    string = string.replace("&gt;", ">")
    return string


""" AWS / CLI / MTurk """

def aws_cli_setup():
    os.environ['AWS_PROFILE'] = 'textmorph'
    os.environ['AWS_CREDENTIAL_PROFILES_FILE'] = '~/.aws/credentials'


def aws_cli_mturk(cmd):
    run = ['aws', '--region', 'us-east-1', 'mturk']
    run = run + cmd
    dat = subprocess.check_output(run)
    return dat


def get_hit_with_id(hit_id):
    cmd = ['get-hit']
    cmd += ['--hit-id'] + [hit_id]
    return json.loads(aws_cli_mturk(cmd))


def delete_hit_with_id(hit_id):  # !!! Permanent deletion from AWS.
    cmd = ['delete-hit']
    cmd += ['--hit-id'] + [hit_id]
    return aws_cli_mturk(cmd)


def get_assignments_for_hit_with_id(hit_id):
    cmd = ['list-assignments-for-hit']
    cmd += ['--hit-id'] + [hit_id]
    return json.loads(aws_cli_mturk(cmd))


def get_reviewable_hit_ids(hit_type_id):
    cmd = ['list-reviewable-hits', '--hit-type-id', hit_type_id]
    rev_hit_ids = []
    rev_hits = json.loads(aws_cli_mturk(cmd))
    while rev_hits['NumResults'] > 0:
        rev_hit_ids.extend(rev_hits['HITs'])
        next_tk = rev_hits['NextToken']
        if next_tk is None:
            break
        else:
            cmd_ = cmd + ['--next-token', next_tk]
            rev_hits = json.loads(aws_cli_mturk(cmd_))
    return [rhi['HITId'] for rhi in rev_hit_ids]


def approve_assignment_with_id(assn_id):
    cmd = ['approve-assignment', '--assignment-id', assn_id]
    return aws_cli_mturk(cmd)


class SimilarityData(object):

    def __init__(self, chains_file):
        self.chains_file = chains_file
        ch, ln = self.load()
        self.chains = ch 
        self.line_numbers = ln

    def load(self):
        chains = []
        line_numbers = []
        with open(self.chains_file, 'r') as chains_f:
            lines = chains_f.readlines()
            peek = lines[1:] + ['']
            temp_chain = []
            for i, (line, next) in enumerate(zip(lines, peek)):
                if line == '\n':
                    chains.append(temp_chain)
                    temp_chain = []
                    continue
                temp_chain.append(line)
                line_numbers.append(i + 1)

        assert_or_report(len(line_numbers) == sum([len(c) for c in chains]),
                         'misformatted params.chains_file')
        print "Loaded {} chains and {} total sentences".format(len(chains),
                                                               sum([len(ch) for ch in chains]))
        return chains, line_numbers

    def _sample_datum(self):
        """
        Sample 1 sentence pair uniformly from a random sentence chain
        """
        assert params.SENT_MAX_DIST > 0

        c_idx = np.random.randint(0, len(self.chains), 1)[0]
        c_len = len(self.chains[c_idx])
        assert c_len > params.SENT_MAX_DIST
        SENT_MAX_DIST_ = min(c_len, params.SENT_MAX_DIST)
        s_idx_1 = np.random.randint(0, c_len - SENT_MAX_DIST_)
        s_idx_2 = np.random.randint(s_idx_1 + 1, s_idx_1 + SENT_MAX_DIST_)
        return c_idx, s_idx_1, s_idx_2

    def sample(self, num_samples):
        """
        Sample num_samples sentence pairs, and return their contents and ids  
        """

        samples = []
        sample_ids = []
        while len(samples) < num_samples:
            c_idx, s_idx_1, s_idx_2 = self._sample_datum()
            s_1_ln = sentence_index_to_line_number(
                    c_idx, s_idx_1, self.chains, self.line_numbers)
            s_2_ln = sentence_index_to_line_number(
                c_idx, s_idx_2, self.chains, self.line_numbers)
            sample_id = params.database.pair_id(s_1_ln, s_2_ln, self.chains_file)
            s_1 = self.chains[c_idx][s_idx_1]
            s_2 = self.chains[c_idx][s_idx_2]
            if s_1 == s_2:
                # no need to deploy
                # save to database and continue
                params.database.add_pair(sample_id, s_1, s_2)
                params.database.add_rating(sample_id, 5, 'AUTO')
                params.database.save()
                continue
            else:
                samples.append((s_1, s_2))
                sample_ids.append(sample_id)

        return samples, sample_ids

    @classmethod
    def format_data_for_template(cls, pairs, pair_ids):
        data = {}
        data['num'] = len(pairs)
        for i, (pair, pair_id) in enumerate(zip(pairs, pair_ids)):
            data['sent_{}_a'.format(i)] = xml_safe(pair[0])
            data['sent_{}_b'.format(i)] = xml_safe(pair[1])
            data['pair_{}_id'.format(i)] = pair_id
        return data 


class CoherenceData:

    def __init__(self, sentences_file):
        self.sentences_file = sentences_file
        st, ln = self.load()
        self.sentences = st 
        self.line_numbers = ln 

    def load(self):
        sentences = []
        line_numbers = []
        with open(self.sentences_file, 'r') as sents_f:
            sentences = [s.strip() for s in sents_f.readlines()]
            line_numbers = range(1, len(sentences) + 1)
        print "Loaded {} sentences".format(len(sentences))
        return sentences, line_numbers

    def sample(self, num_samples):
        s_idxs = np.random.randint(0, len(self.sentences), num_samples)
        samples = [self.sentences[idx] for idx in s_idxs]
        sample_ids = [MTurkData.sent_id(self.line_numbers[idx], 
                                        self.sentences_file) for idx in s_idxs]
        return samples, sample_ids 
        
    def format_data_for_template(self, sentences, sentence_ids):
        data = {}
        data['num'] = len(sentences)
        data['fname'] = params.database.sent_id("", self.sentences_file)
        for i, (s, s_id) in enumerate(zip(sentences, sentence_ids)):
            data['sent_{}'.format(i)] = xml_safe(s)
            data['sent_{}_id'.format(i)] = s_id
        return data 


class SimilarityHIT(object):

    def __init__(self, hit_id, hit_data):
        self.id = hit_id
        self.questions = SimilarityHIT.get_questions(hit_data)
        assignments = get_assignments_for_hit_with_id(hit_id)
        assignments = assignments['Assignments']
        self.is_complete = len(assignments) > 0
        self.answers = []
        if self.is_complete:
            assn = assignments[0]
            self.assignment = assn
            self.assignment_id = assn['AssignmentId']
            self.assignment_status = assn['AssignmentStatus']
            self.worker = assn['WorkerId']
            self.answers = SimilarityHIT.get_answers(assn)

    @classmethod
    def get(cls, hit_id):
        hit_data = get_hit_with_id(hit_id)
        return SimilarityHIT(hit_id, hit_data)

    @classmethod
    def get_questions(cls, hit_data):
        qs = hit_data['HIT']['Question']
        qid = "<QuestionIdentifier>(.*?)<\/QuestionIdentifier>"
        qttl = "<Title>.*?</Title>"
        qtxt = "<Text>(.*?)<\/Text><Text>(.*?)<\/Text>"
        rgx = "{}.*?<QuestionContent>{}.*?{}<\/QuestionContent>".format(
            qid, qttl, qtxt)
        questions = re.findall(rgx, qs, flags=re.DOTALL)
        questions = {p_id: (q_1, q_2) for (p_id, q_1, q_2) in questions}
        return questions

    @classmethod
    def get_answers(cls, assignment):
        ans = assignment['Answer']
        qi = "<QuestionIdentifier>(.*?_.*?)<\/QuestionIdentifier>"
        si = "<SelectionIdentifier>([0-9])<\/SelectionIdentifier>"
        rgx = "<Answer>\n{}\n{}\n<\/Answer>".format(qi, si)
        answers = re.findall(rgx, ans)
        answers = {p_id: ans for (p_id, ans) in answers}
        return answers

    def print_pretty(self):
        for p_id in self.questions:
            q_1, q_2 = self.questions[p_id]
            q_1 = un_xml_safe(q_1).strip()
            q_2 = un_xml_safe(q_2).strip()
            a = self.answers[p_id]
            print "Q1: {}\nQ2: {}\nRATE: {}".format(q_1, q_2, a)

    def save(self, database):
        for pair_id, (s_1, s_2) in self.questions.iteritems():
            rate = hit.answers[pair_id]
            database.add_pair(pair_id, s_1, s_2)
            database.add_rating(pair_id, int(rate), self.worker)
        database.complete_hit(self.id)


class CoherenceHIT:

    def __init__(self, hit_id, hit_data):
        self.id = hit_id
        self.questions = CoherenceHIT.get_questions(hit_data)
        assignments = get_assignments_for_hit_with_id(hit_id)
        assignments = assignments['Assignments']
        self.is_complete = len(assignments) > 0
        self.answers = []
        if self.is_complete:
            assn = assignments[0]
            self.assignment = assn
            self.assignment_id = assn['AssignmentId']
            self.assignment_status = assn['AssignmentStatus']
            self.worker = assn['WorkerId']
            self.answers = CoherenceHIT.get_answers(assn)

    @classmethod 
    def get(cls, hit_id):
        hit_data = get_hit_with_id(hit_id)
        return CoherenceHIT(hit_id, hit_data)

    @classmethod
    def get_questions(cls, hit_data):
        qs = hit_data['HIT']['Question']
        qid = "<QuestionIdentifier>(.*?)_[c|g]<\/QuestionIdentifier>"
        qttl = "<Title>.*?<\/Title>"
        qtxt = "<FormattedContent><!\[CDATA\[<b>(.*?)<\/b>\]\]><\/FormattedContent>"
        rgx = "{}.*?<QuestionContent>{}.*?{}.*?<\/QuestionContent>".format(
            qid, qttl, qtxt)
        questions = re.findall(rgx, qs, flags=re.DOTALL)
        questions = {p_id: s for (p_id, s) in questions}
        return questions

    @classmethod
    def get_answers(cls, assignment):
        ans = assignment['Answer']
        qi = "<QuestionIdentifier>(.*?)_([c|g])<\/QuestionIdentifier>"
        qi_sim = "<QuestionIdentifier>sim_(.*?)_<\/QuestionIdentifier>"
        si = "<SelectionIdentifier>([0-9])<\/SelectionIdentifier>"
        rgx = "<Answer>\n{}\n{}\n<\/Answer>".format(qi, si)
        rgx_sim = "<Answer>\n{}\n{}\n<\/Answer>".format(qi_sim, si)

        answers = defaultdict(dict)
        for (q, t, a) in re.findall(rgx, ans):
            answers[q][t] = a

        sim_id, sim = re.findall(rgx_sim, ans)[0]
        answers['sim'] = (sim_id, sim)

        return answers

    def print_pretty(self):
        pass

    def save(self, database):
        for sent_id, s in self.questions.iteritems():
            sc_c = self.answers[sent_id]['c']
            sc_g = self.answers[sent_id]['g']
            database.add_sent(sent_id, s)
            database.add_sent_scores(sent_id, int(sc_c), int(sc_g), self.worker)
        sim_id, sim_sc = self.answers['sim']
        database.add_sent_sim_score(sim_id, int(sim_sc), self.worker)
        database.complete_hit(self.id)

""" main functions """

def put(task_obj, data_obj):

    num_datum = max(params.num_datum, params.DATUM_PER_EXP)
    num_expms = int(math.ceil(float(num_datum) / params.DATUM_PER_EXP))
    num_datum = num_expms * params.DATUM_PER_EXP

    curr_samples, curr_sample_ids = [], []
    for exp in range(num_expms):

        try:
            samples, sample_ids = data_obj.sample(params.DATUM_PER_EXP)
            curr_samples = samples
            curr_sample_ids = sample_ids

            samples_formatted = data_obj.format_data_for_template(samples, sample_ids)
            hit_id = task_obj.launch(samples_formatted)
            assert_or_report(hit_id is not None,
                             "Invalid HIT ID for: {data}".format(data=samples_formatted))
            print "Deployed HIT: {} ({}/{})".format(hit_id, exp + 1, num_expms)

            # save HIT to DB 
            params.database.add_hit(hit_id)

            # wait before next request to MTurk
            time.sleep(0.25)

        except Exception, e:
            msg = "Failed to deploy HIT: {err}, {data}".format(err=e, data=curr_samples)
            print msg 
            print traceback.format_exc()

        # allow opportunity to cancel
        if exp == 0 and not user_confirmation("Deployed 1 task, deploy remainder? (y/n)"):
            print "cancelled"
            break

    params.database.save()


def get(hit_obj):

    # get HITs listed as incomplete
    inc_hit_ids = params.database.get_incomplete_hits()
    print "Ok. Loaded {} incomplete HITs from data store".format(len(inc_hit_ids))

    # query answers from MTurk
    mturk_hits = [hit_obj.get(h_id) for h_id in inc_hit_ids]
    num_cmpl = len([h for h in mturk_hits if h.is_complete])
    print "{} of these have been completed.".format(num_cmpl)

    if user_confirmation("Save? (y/n)"):
        # update ratings
        for hit in mturk_hits:
            if hit.is_complete:
                hit.save(params.database)

        params.database.save()


def review():
    s = SimilarityTask(params.debug)

    reviewable_hits = []
    load_from_db = user_confirmation(
        "load completed assignments from (d)atabase, or from (r)eviewable_hits?", yes="d")
    if load_from_db:
        reviewable_hits = params.database.get_complete_hits()
    else:
        reviewable_hits = get_reviewable_hit_ids(s.hit_type_id)
    print "Loading {} HITs for review".format(len(reviewable_hits))

    workers_to_hits = defaultdict(list)
    for rh_id in reviewable_hits:
        hit = get_hit(rh_id)
        worker_id = hit['worker']
        workers_to_hits[worker_id].append(hit)

    workers_to_review = [w_id for (w_id, hits) in workers_to_hits.iteritems(
    ) if any([h['AssignmentStatus'] != 'Approved' for h in hits])]
    blacklist = []
    if user_confirmation("Are you ready to review? ({} workers) (y/n)".format(len(workers_to_review))):
        for i, worker_id in enumerate(workers_to_review):
            hits = workers_to_hits[worker_id]
            print "Reviewing worker: {} ({}/{})".format(worker_id, i + 1, len(workers_to_review))
            print_pretty_assignment(hits[0])
            if user_confirmation("approve all HITs for this worker? (y/n)"):
                for hit in hits:
                    assn_id = hit['AssignmentId']
                    assn_st = hit['AssignmentStatus']
                    if assn_st != "Approved":
                        print "-- approving H_ID:{}, A_ID:{}".format(hit['Id'], assn_id)
                        approve_assignment_with_id(assn_id)
                print "approved {} hits".format(len(hits))
            else:
                blacklist.append(worker_id)

    print "blacklist:", blacklist


if __name__ == '__main__':

    opt = parse_args(sys.argv[1:])

    print "received params:"
    print "\t debug:{}".format(params.debug)

    aws_cli_setup()
    params.database = MTurkData(params.database_file)

    task_obj = None 
    data_obj = None
    hit_obj = None 
    if params.task_type == params.TASK_SIMILARITY:
        if opt == 'put':
            task_obj = SimilarityTask(params.debug)
            data_obj = SimilarityData(params.data_file)
        hit_obj = SimilarityHIT
    elif params.task_type == params.TASK_COHERENCE:
        if opt == 'put':
            task_obj = CoherenceTask(params.debug)
            data_obj = CoherenceData(params.data_file)
        hit_obj = CoherenceHIT
    else:
        assert False 

    if opt == 'put':
        put(task_obj, data_obj)
    if opt == 'get':
        get(hit_obj)
    if opt == 'review':
        review()

