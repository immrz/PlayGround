import torch
import traceback


class PartialParse:
    def __init__(self, ex):
        self.ex = ex
        self.n_words = len(ex['word']) - 1
        self.stack = [0]
        self.buf = [i + 1 for i in range(self.n_words)]
        self.arcs = []
        self.success = 0

    def parse_step(self, transition, parser):
        legal_labels = parser.legal_labels(self.stack, self.buf)
        if legal_labels[transition] != 1:
            wrong_step = 'S'
            if transition < parser.n_deprel:
                wrong_step = 'L'
            elif transition < parser.n_deprel * 2:
                wrong_step = 'R'
            raise ValueError('Illegal transition step! Stack length: {:d}; Buffer length: {:d}; '
                             'Transition: {:s}'.format(len(self.stack), len(self.buf), wrong_step))

        if transition == parser.n_trans - 1:  # shift
            self.stack.append(self.buf[0])
            self.buf = self.buf[1:]
        elif transition < parser.n_deprel:  # left arc
            self.arcs.append((self.stack[-1], self.stack[-2], transition))
            self.stack = self.stack[:-2] + [self.stack[-1]]
        else:  # right arc
            self.arcs.append((self.stack[-2], self.stack[-1], transition - parser.n_deprel))
            self.stack = self.stack[:-1]

    def parse(self, model, parser):
        for i in range(self.n_words * 2):
            feature = parser.extract_features(self.stack, self.buf, self.arcs, self.ex)
            legal_label = parser.legal_labels(self.stack, self.buf)
            prob = model(torch.LongTensor(feature), torch.DoubleTensor(legal_label).view(1, -1))
            transition = torch.argmax(prob, dim=-1).item()

            try:
                self.parse_step(transition, parser)
            except ValueError:
                break

        else:
            self.success = 1

    def safe_parse(self, model, parser):
        for i in range(self.n_words * 2):
            feature = parser.extract_features(self.stack, self.buf, self.arcs, self.ex)
            legal_label = parser.legal_labels(self.stack, self.buf)
            prob = model(torch.LongTensor(feature), torch.DoubleTensor(legal_label).view(1, -1))

            # choose the argmax in legal labels
            desc_idx = torch.argsort(prob.squeeze(), dim=-1, descending=True).tolist()
            transition = desc_idx[0]
            for transition in desc_idx:
                if legal_label[transition] > 0:
                    break
            self.parse_step(transition, parser)

        self.success = 1

    def accuracy(self, unlabeled=True):
        if not self.success:
            return 0
        sorted_arcs = sorted(self.arcs, key=lambda a: a[1])
        if unlabeled:
            gt = self.ex['head'][1:]
            pred = [a[0] for a in sorted_arcs]
            num_correct = sum([1 if gt[i] == pred[i] else 0 for i in range(len(gt))])
        else:
            gt_head = self.ex['head'][1:]
            gt_label = self.ex['label'][1:]
            pred_head = [a[0] for a in sorted_arcs]
            pred_label = [a[2] for a in sorted_arcs]
            num_correct = sum([1 if gt_head[i] == pred_head[i] and gt_label[i] == pred_label[i] else 0
                               for i in range(len(gt_head))])
        return num_correct, self.n_words
