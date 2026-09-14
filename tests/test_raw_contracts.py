import unittest
from dataclasses import asdict
from h65.raw.contracts import (EpisodePublic, preview_timeline, proposals, physical_uniform,
    selection_from_support, swap, tubelets, repartition, project_ground_truth, geometric_swaps)


def episode(valid=768):
    ids = tuple(i*4 for i in range(valid))+(4*(valid-1),)*(768-valid)
    return EpisodePublic('synthetic',0,'development','unused.mp4',25.,4000,160.,0,4,ids,
                         tuple(i < valid for i in range(768)))


class RawContractsTest(unittest.TestCase):
    def test_domain_and_uniform(self):
        ep = episode()
        preview = preview_timeline(ep)
        official, raw = (proposals(ep,preview,domain) for domain in ('O','R+'))
        self.assertTrue(set(official.frame_ids) < set(raw.frame_ids))
        self.assertLessEqual(len(raw.frame_ids),768+192*4)
        self.assertEqual(len(preview.frame_ids),192)
        for p in (official,raw):
            selection = physical_uniform(ep,p)
            self.assertEqual(len(selection.support),384)
            self.assertTrue(set(selection.support) <= set(p.frame_ids))
            self.assertEqual(physical_uniform(ep,p),selection)
            self.assertEqual(len(tubelets(selection,ep.fps).contributor_pairs),192)
        self.assertNotIn('gt_segments',asdict(ep))

    def test_tail_validity_is_shared(self):
        ep = episode(11)
        preview = preview_timeline(ep)
        for domain in ('O','R+'):
            selection = physical_uniform(ep,proposals(ep,preview,domain))
            self.assertEqual(sum(selection.valid),11)
            self.assertEqual(len(selection.frame_ids),384)
            self.assertEqual(tubelets(selection,ep.fps).contributor_valid[5],(True,False))
        self.assertLess(len(set(preview.frame_ids)),192)

    def test_swap_repartitions_whole_interval(self):
        before = selection_from_support(range(0,32,2),16)
        ep = episode()
        p = proposals(ep,preview_timeline(ep),'R+')
        insert = next(i for i in p.frame_ids if 30 < i < 65 and i not in before.support)
        after = swap(before,0,insert,p)
        change = repartition(before,after,25.)
        self.assertGreater(change['n_changed_pairs'],1)
        self.assertEqual(change['n_changed_packs'],1)
        self.assertEqual(len(change['delta_pair_span_seconds']),8)
        with self.assertRaises(ValueError):
            swap(before,0,2,p)

    def test_projection_and_noop(self):
        ep = episode(11)
        gt = project_ground_truth(ep,[dict(segment=[-1,1.],label='A'),dict(segment=[1.,5.],label='A')],['A'])
        self.assertEqual(gt['gt_boundary_validity'],[(False,True),(True,False)])
        self.assertEqual(gt['gt_segments'],[(0.,6.25),(6.25,10.)])
        p = proposals(ep,preview_timeline(ep),'R+')
        selection = physical_uniform(ep,p)
        self.assertEqual(repartition(selection,selection,ep.fps)['n_changed_pairs'],0)
        for a,b in geometric_swaps(ep,p,selection):
            self.assertEqual(sum(swap(selection,a,b,p).valid),sum(selection.valid))


if __name__ == '__main__':
    unittest.main()
