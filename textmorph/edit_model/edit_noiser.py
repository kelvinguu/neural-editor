import numpy as np

from textmorph.edit_model.editor import EditExample


class EditNoiser(object):

    def __init__(self, ident_pr = 0.1, attend_pr = 0.5):
        self.ident_pr = ident_pr
        self.attend_pr = attend_pr

    def __call__(self, examples):
        """Return a batch of noisy EditExamples.

        Does not modify the original EditExamples.
        """
        return [self._noise(ex) for ex in examples]

    def dropout_split(self, word_list):
        pr_list = [1.0-self.attend_pr, self.attend_pr]
        if len(word_list)>0:
            num_sampled = np.random.choice(np.arange(len(pr_list)), 1, p=pr_list)
            num_sampled = min(num_sampled, len(word_list))
            choice_index = np.random.choice(np.arange(len(word_list)), num_sampled, replace=False)
            mask = np.zeros(len(word_list), dtype=bool)
            mask[choice_index] = True
            warray = np.array(word_list)
            return (warray[mask]).tolist(), (warray[np.invert(mask)]).tolist()
        else:
            return [], []

    def _noise(self, ex):
        """Return a noisy EditExample.

        Note: this strategy is only appropriate for diff-style EditExamples.

        Args:
            ex (EditExample)

        Returns:
            EditExample: a new example. Does not modify the original example.
        """
        ident_map = np.random.binomial(1,self.ident_pr)
        if ident_map:
            return EditExample(ex.source_words, [], [], [], [], ex.source_words)
        else:
            insert_exact, insert_approx= self.dropout_split(ex.insert_exact_words)
            delete_exact, delete_approx = self.dropout_split(ex.delete_exact_words)
            return EditExample(ex.source_words, insert_approx, insert_exact, delete_approx, delete_exact, ex.target_words)
