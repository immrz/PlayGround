from depparse.trans_based.parse import PartialParse
import torch


_row_format = '{:10s}{:6s}{:20s}{:6s}{:20s}'
_first_row = _row_format.format('WORD', 'HEAD', 'LABEL', 'PHEAD', 'PLABEL')


def print_fault(example, parser, fo, res_arcs=None):
    word = [parser.id2tok[i] for i in example['word'][1:]]
    head = [str(i) for i in example['head'][1:]]
    label = [parser.id2tok[i] if i in parser.id2tok else '<UNK>' for i in example['label'][1:]]
    assert len(word) == len(head) and len(word) == len(label)

    if res_arcs is None:
        make_up = ['' for _ in range(len(word))]
        output = zip(word, head, label, make_up, make_up)
    else:
        assert len(word) == len(res_arcs)
        sorted_arcs = sorted(res_arcs, key=lambda t: t[1])
        p_head = [str(t[0]) for t in sorted_arcs]
        p_label = [parser.id2tok[t[2]] if t[2] in parser.id2tok else '<UNK>' for t in sorted_arcs]
        output = zip(word, head, label, p_head, p_label)

    output = [_row_format.format(*list(row)) for row in output]
    fo.write('\n'.join(output) + '\n\n')


def parse_and_output(model, data, parser, fo):
    n_ex = 0
    n_match = 0
    for example in data:
        ex = PartialParse(example)
        ex.safe_parse(model, parser)
        if not ex.success:
            print_fault(example, parser, fo)
        else:
            correct, n_words = ex.accuracy(parser.unlabeled)
            if correct < n_words:
                print_fault(example, parser, fo, res_arcs=ex.arcs)
            else:
                n_match += 1
        n_ex += 1

    print('Output / Total: {:d} / {:d}'.format(n_ex - n_match, n_ex))


def diagnosing(model, data, parser, output_file, **kwargs):
    model.eval()
    with torch.set_grad_enabled(False), open(output_file, 'w') as fo:
        fo.write(_first_row + '\n')
        parse_and_output(model, data, parser, fo)
