import json
from string import Template

import boto
from boto.mturk.question import Overview, QuestionContent, SelectionAnswer, Question, AnswerSpecification, QuestionForm

from gtd.turk import Task, get_mturk_connection, standard_quals
from gtd.utils import Config
from textmorph import data


"""
To review completed HITs:
- Go to: https://requester.mturk.com/mturk/manageHITs

To do a HIT:
- Go to: https://worker.mturk.com/
- Search for "percy liang"
- Click "Accept & Work"
  - For some reason, I had trouble viewing these HITs on Google Chrome (invalid URL parameter error).
  - On Firefox, things are fine.
"""

config = Config.from_file(data.workspace.config)
mtc = get_mturk_connection(config.aws_access_key_id,
                           config.aws_secret_access_key, sandbox=False)


class SimilarityTask(Task):

    def __init__(self, debug):
        # load from configuration
        conf = Config.from_file(data.workspace.turk.similarity.config.txt)
        self.title = conf.title
        self.description = conf.description
        self.keywords = conf.keywords
        self.price = conf.price
        self.duration = eval(conf.duration)
        self.approval_delay = eval(conf.approval_delay)

        # store form specification as JSON, to be built automatically on launch
        with open(data.workspace.turk.similarity.form.json) as form_json:
            self.form_json = form_json.read()

        price_per_hit = 0.0 if debug else self.price

        quals = standard_quals(debug)

        hit_type_ids = mtc.register_hit_type(title=self.title, description=self.description, reward=price_per_hit,
                                             duration=self.duration,
                                             keywords=self.keywords, approval_delay=self.approval_delay, qual_req=quals)
        hit_type_id = hit_type_ids[0].HITTypeId

        super(SimilarityTask, self).__init__(hit_type_id, mtc)

    def launch(self, data={}):
        qf = QuestionForm()
        form_json = BotoFormGenerator.inject_data(self.form_json, data)
        BotoFormGenerator.from_json(qf, form_json)
        return self.create_hit(qf)


class CoherenceTask(Task):

    def __init__(self, debug):
        # load from configuration
        conf = Config.from_file(data.workspace.turk.coherence.config.txt)
        self.title = conf.title
        self.description = conf.description
        self.keywords = conf.keywords
        self.price = conf.price
        self.duration = eval(conf.duration)
        self.approval_delay = eval(conf.approval_delay)

        # store form specification as JSON, to be built automatically on launch
        with open(data.workspace.turk.coherence.form.json) as form_json:
            self.form_json = form_json.read()

        price_per_hit = 0.0 if debug else self.price

        quals = standard_quals(debug)

        hit_type_ids = mtc.register_hit_type(title=self.title, description=self.description, reward=price_per_hit,
                                             duration=self.duration,
                                             keywords=self.keywords, approval_delay=self.approval_delay, qual_req=quals)
        hit_type_id = hit_type_ids[0].HITTypeId

        super(CoherenceTask, self).__init__(hit_type_id, mtc)

    def launch(self, data={}):
        qf = QuestionForm()
        form_json = BotoFormGenerator.inject_data(self.form_json, data)
        BotoFormGenerator.from_json(qf, form_json)
        return self.create_hit(qf)


class BotoFormGenerator(object):

    form_types = {'Overview', 'QuestionContent', 'SelectionAnswer', 'Question', 'AnswerSpecification', 'QuestionForm', 'FormattedContent'}

    @staticmethod
    def from_json(question_form, json_data):
        """
        Construct a QuestionForm from a JSON specification
        """

        form_data = json.loads(json_data, strict=False)

        # construct objects and build QuestionForm
        for obj_data in form_data['form']:
            obj = BotoFormGenerator._from_data(obj_data)
            question_form.append(obj)

    @staticmethod
    def _from_data(form_data):
        """
        Generates and populates boto.mturk.question objects from a specification.
        """

        if type(form_data) is not dict:
            return form_data

        """
        Functions for creating form objects. 
        args_dict is a dictionary containing a mapping from names to arguments. 
        Positional and keyword arguments pertaining to the particular object
        are extracted from args_dict and passed appropriately to the object 
        constructor.

        It's very easy to add functionality to this scheme. Simply add a form_type
        and a make_{} function with the correct required args_, and it can 
        immediately be used in the JSON spec.
        """
        def make_args(args_dict, args_):
            # positional arguments
            args = [args_dict[k] for k in args_]
            # keyword arguments
            kwargs = {k: v for k, v in args_dict.iteritems() if k not in args_}
            return args, kwargs

        def add_field(obj, field):
            (fl_name, fl_value) = next(field.iteritems())
            obj.append_field(fl_name, fl_value)

        def add_append(obj, append):
            obj.append(append)

        def make_Overview(args_dict, args_=[]):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.Overview(*args, **kwargs)

        def make_Question(args_dict, args_=['identifier', 'content', 'answer_spec']):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.Question(*args, **kwargs)

        def make_QuestionContent(args_dict, args_=[]):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.QuestionContent(*args, **kwargs)

        def make_SelectionAnswer(args_dict, args_=[]):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.SelectionAnswer(*args, **kwargs)

        def make_AnswerSpecification(args_dict, args_=['spec']):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.AnswerSpecification(*args, **kwargs)

        def make_FormattedContent(args_dict, args_=['content']):
            args, kwargs = make_args(args_dict, args_)
            return boto.mturk.question.FormattedContent(*args, **kwargs)

        k, v = next(form_data.iteritems())
        if k in BotoFormGenerator.form_types:
            make_fn = eval("make_{}".format(k))
            args = {}  # arguments to the object, that may be other objects
            # list of to be appended form objects (Field-type or otherwise
            fields = []

            for arg_k, arg_v in v.iteritems():  # iterate over arguments to the form object
                # Fields _or_ form objects to be appended (e.g.
                # FormattedContent)
                if arg_k == "fields":
                    fields = arg_v
                else:  # recurse and build form object argument
                    args[arg_k] = BotoFormGenerator._from_data(arg_v)

            obj = make_fn(args)
            for fl in fields:
                fl_k, fl_v = next(fl.iteritems())
                if fl_k == "field":
                    add_field(obj, fl_v)
                if fl_k == "append":
                    ap = BotoFormGenerator._from_data(fl_v)
                    add_append(obj, ap)
            return obj

        return None

    @staticmethod
    def inject_data(json_data, data):
        """
        Insert data into the JSON format specification.
        This is used to dynamically create forms with different questions using
        the same specification.
        """
        return Template(json_data).substitute(**data)
