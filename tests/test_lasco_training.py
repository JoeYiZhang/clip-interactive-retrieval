import unittest
import numpy as np
import torch
from cir_baseline.lasco_training import prepare_examples, candidate_masks
from cir_baseline.learned_fusion import masked_multi_positive_loss

def row(target, text="blue", qid=7):
    return {"qid":qid, "query-text":text, "query-image":[1,"train2014/COCO_train2014_000000000001.jpg"],
            "target-image":[target,f"train2014/COCO_train2014_{target:012}.jpg"]}

class LaSCoTrainingTests(unittest.TestCase):
    def test_row_identity_survives_source_qid_reuse_and_quarantine(self):
        examples, _, bad = prepare_examples([row(2),row(3," "),row(4)], "train")
        self.assertEqual([e['example_id'] for e in examples],['lasco:train:0','lasco:train:2'])
        self.assertEqual(bad[0]['example_id'],'lasco:train:1')

    def test_known_positives_are_condition_specific_and_reference_excluded(self):
        examples, known, _ = prepare_examples([row(2),row(3," blue "),row(4,"red")],"train")
        positive, valid = candidate_masks(examples,[1,2,3,4],known)
        np.testing.assert_array_equal(positive[0],[False,True,True,False])
        np.testing.assert_array_equal(positive[2],[False,False,False,True])
        self.assertFalse(valid[:,0].any())
        with self.assertRaises(ValueError):
            candidate_masks(examples,[1,2,2,4],known)

    def test_masked_reference_cannot_affect_loss_or_receive_gradient(self):
        query = torch.tensor([[1.,0.]],requires_grad=True)
        candidate = torch.tensor([[100.,0.],[1.,0.],[0.,1.]],requires_grad=True)
        positive = torch.tensor([[False,True,False]])
        valid = torch.tensor([[False,True,True]])
        loss = masked_multi_positive_loss(query,candidate,positive,valid)
        loss.backward()
        torch.testing.assert_close(candidate.grad[0],torch.zeros(2))
        changed = candidate.detach().clone()
        changed[0] = torch.tensor([-100.,50.])
        torch.testing.assert_close(loss,masked_multi_positive_loss(query,changed,positive,valid))
        with self.assertRaises(ValueError):
            masked_multi_positive_loss(query,candidate,positive,torch.zeros_like(valid))

    def test_wrong_split_or_image_id_path_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_examples([row(2)],'val')
