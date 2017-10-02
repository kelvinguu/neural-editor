import copy 
import random
import sys
import urllib
from abc import ABCMeta
from collections import defaultdict

import boto.mturk.qualification as mtqu
from boto.mturk.connection import MTurkConnection, MTurkRequestError
from boto.mturk.question import ExternalQuestion

from gtd.chrono import verboserate
from gtd.utils import chunks, SimpleExecutor, Failure, parallel_call


# Connect to MTurk
def get_mturk_connection(aws_access_key_id, aws_secret_access_key, sandbox=False):
    host = 'mechanicalturk.amazonaws.com'
    if sandbox:
        host = 'mechanicalturk.sandbox.amazonaws.com'
    mtc = MTurkConnection(aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          host=host)
    return mtc


def standard_quals(debug):
    """Construct qualifications for MTurk task.

    Args:
        debug (bool): if True, use a special qualification that only we possess.
    """
    quals = mtqu.Qualifications()
    if debug:
        quals.add(mtqu.Requirement("3B9TX86P8NTZUJU62N2SLJ1DZCERL5", "EqualTo", 100, required_to_preview=True))
    else:
        quals.add(mtqu.LocaleRequirement("EqualTo", "US", required_to_preview=True))
        quals.add(mtqu.PercentAssignmentsApprovedRequirement("GreaterThan", 95))
        quals.add(mtqu.NumberHitsApprovedRequirement("GreaterThan", 10))
    return quals


def _get_all_hits(get_page):
    """Given a function that retrieves a single page of HITs, retrieve all HITs.

    WARNING:
        - this function can be quite slow.
        - results are returned in no particular order.

    Args:
        get_page (Callable[[int, int], list[HIT]]): a function which takes a page size and page number.
            and returns a list of HITs.

            kwargs:
                page_size (int)
                page_number (int)

    Returns:
        generator[HIT]
    """
    page_size = 100  # HITs per page

    # compute the pages that need to be fetched
    search_results = get_page(page_size=page_size, page_number=1)
    total_hits = int(search_results.TotalNumResults)
    total_pages = total_hits / page_size + bool(total_hits % page_size)
    page_nums = list(range(1, total_pages + 1))

    # fetch all the pages in parallel
    fetch_page = lambda i: get_page(page_size=page_size, page_number=i)
    with SimpleExecutor(fetch_page) as executor:
        for i in page_nums:
            executor.submit(i, i)
        for i, page in verboserate(executor.results(), desc='Fetching pages of HITs', total=total_pages):
            if isinstance(page, Failure):
                print page.traceback
                continue
            for hit in page:
                yield hit


class Task(object):
    __metaclass__ = ABCMeta

    def __init__(self, hit_type_id, mtc):
        """Construct task.

        Args:
            hit_type_id (HITTypeId)
            mtc (MTurkConnection)
        """
        self._hit_type_id = hit_type_id
        self._mtc = mtc

    def create_hit(self, question):
        """Create a HIT with the specified Question.

        Args:
            question (QuestionForm|HTMLQuestion|ExternalQuestion)

        Returns:
            str: HITId
        """
        lifetime = 3600 * 24 * 30  # HIT will be shown on the marketplace for 30 days total
        hits = self._mtc.create_hit(
            hit_type=self._hit_type_id,
            question=question,
            lifetime=lifetime,
            max_assignments=1
        )
        hit = hits[0]
        return hit.HITId

    @property
    def hit_type_id(self):
        """The HIT type ID shared by all HITs of this task."""
        return self._hit_type_id

    def get_hits(self):
        """Return a generator over all HITs for this task.

        WARNING: you will occasionally get AWS no-service errors.
        """
        get_page = lambda page_size, page_number: self._mtc.search_hits(sort_direction='Descending', page_size=page_size,
                                                                  page_number=page_number)
        hit_type_id = self._hit_type_id

        for hit in _get_all_hits(get_page):
           if hit.HITTypeId == hit_type_id:
                yield hit

    def get_reviewable_hits(self):
        """Return a generator over all Reviewable HITs associated with this task.

        A HIT is reviewable if there are assignments pending review.
        TODO(kelvin): not sure if this is correct
        """
        hit_type_id = self._hit_type_id
        get_page = lambda page_size, page_number: self._mtc.get_reviewable_hits(sort_direction='Descending',
                                                                          hit_type=hit_type_id,
                                                                          page_size=page_size,
                                                                          page_number=page_number)
        return _get_all_hits(get_page)

    def expire_hits(self):
        """Expire all HITs associated with this task.

        The task will no longer appear in the marketplace.
        But the HITs and their assignments will still exist on MTurk, awaiting approval/rejection.
        """
        hits = list(self.get_hits())
        hit_ids = [hit.HITId for hit in hits]
        parallel_call(self._mtc.expire_hit, hit_ids)

    def disable_hits(self):
        """Disable all HITs associated with this task.

        WARNING: approves all submitted assignments that have not already been approved or rejected.
        """
        hits = list(self.get_hits())
        hit_ids = [hit.HITId for hit in hits]
        parallel_call(self._mtc.disable_hit, hit_ids)

    def report_progress(self):
        completed = len(list(self.get_reviewable_hits()))
        total = len(list(self.get_hits()))
        print '{}/{} complete'.format(completed, total)

    def get_all_assignments(self, hits=[]):
        assignments = []
        assignments_batch = parallel_call(self._mtc.get_assignments, [hit.HITId for hit in hits])
        for assignments in assignments_batch:
            for assignment in assignments:
                assignments.append(assignments)
        return assignments

    def review_hits(self, print_assignment):
        """Interactively reject/approve all HITs associated with this task."""
        hits = list(self.get_reviewable_hits())
        worker_to_assignments = defaultdict(list)
        assignments_batch = parallel_call(self._mtc.get_assignments, [hit.HITId for hit in hits])
        for assignments in assignments_batch:
            for assignment in assignments:
                worker_to_assignments[assignment.WorkerId].append(assignment)

        def approve(assignment):
            self._mtc.approve_assignment(assignment.AssignmentId)
            try:
                self._mtc.dispose_hit(assignment.HITId)
            except MTurkRequestError:
                print 'Failed to dispose HIT {}'.format(assignment.HITId)
                raise

        total_workers = len(worker_to_assignments)
        for i, (worker, assignments) in enumerate(worker_to_assignments.items()):
            print "Answers of worker {} ({} of {}, completed {} HITs):".format(worker, i+1, total_workers,
                                                                               len(assignments))

            while True:
                assignment = random.choice(assignments)
                print_assignment(assignment)

                answer = self._prompt_yes_no_more()
                if answer == "y":
                    parallel_call(approve, assignments)
                    print "Approved all assignments for this worker"
                    break
                elif answer == "n":
                    print "Did not approve assignments for this worker"
                    break
            print "\n----- ----- ----- ----- ----- ----- ----- ----- ----- ----- \n"

    def _prompt_yes_no_more(self):
        """Ask a [y]es/[n]o/[s]ee more question via raw_input() and return their answer.

        The "answer" return value is "y" for yes, "n" for no and "s" to see more.
        """
        valid = ["y", "n", "m"]
        while True:
            sys.stdout.write("Approve worker [y/n] or see [m]ore?")
            choice = raw_input().lower()
            if choice in valid:
                return choice
            else:
                sys.stdout.write("Please respond with 'y' or 'n' or 'm'\n")


class ExternalQuestionTask(Task):
    def __init__(self, url, batch_size, hit_type_id, price_per_hit, mtc):
        """Construct task.

        Args:
            url (str): URL where the task is hosted
            batch_size (int): number of examples to present per HIT
            hit_type_id (HITTypeId)
            price_per_hit (float): in dollars
            mtc (MTurkConnection)
        """
        self._url = url
        self._batch_size = batch_size
        self._price_per_hit = price_per_hit
        super(ExternalQuestionTask, self).__init__(hit_type_id, mtc)

    def url(self):
        """The task URL, not including parameters appended to the end (str)."""
        return self._url

    def launch(self, example_uids):
        """Launch task.

        Args:
            example_uids (list[str]): list of example_uids to launch the task with

        """
        batches = list(chunks(example_uids, self._batch_size))

        total_hits = len(batches)
        assert isinstance(self._price_per_hit, float)
        total_cost = total_hits * self._price_per_hit

        print 'Launching {} HITs (${}). Type Enter to continue.'.format(total_hits, total_cost)
        raw_input()

        parallel_call(self.create_hit, batches)

    def create_hit(self, ex_uids):
        """Create a HIT involving the specified Examples.

        Args:
            ex_uids (list[str]): a list of Example UIDs

        Returns:
            str: HITId
        """
        param_str = urllib.urlencode({'exampleUIDs': ','.join(ex_uids)})
        custom_url = '?'.join([self._url, param_str])
        external_question = ExternalQuestion(custom_url, frame_height=500)
        return super(self, ExternalQuestionTask).create_hit(external_question)