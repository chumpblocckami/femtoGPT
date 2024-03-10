import unittest
from gpt2_femto import gelu, softmax, layer_norm, linear, ffn, attention, mha, transformer_block, gpt2, generate, main

class TestGPT2(unittest.TestCase):

    def test_gelu(self):
        self.assertAlmostEqual(gelu(1), 0.841191, places=6)

    def test_softmax(self):
        self.assertAlmostEqual(softmax([1, 2, 3]), [0.09003057317038046, 0.24472847105479767, 0.6652409557748219], places=6)

    def test_layer_norm(self):
        self.assertAlmostEqual(layer_norm([1, 2, 3], 1, 0), [-1.224744871391589, 0.0, 1.224744871391589], places=6)

    def test_linear(self):
        self.assertAlmostEqual(linear([1, 2, 3], [[1, 2, 3], [4, 5, 6]], 1), [15, 33], places=6)

    def test_ffn(self):
        self.assertAlmostEqual(ffn([1, 2, 3], {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}), [61, 143], places=6)

    def test_attention(self):
        self.assertAlmostEqual(attention([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[0, -1e10], [0, 0]]), [0, 150], places=6)

    def test_mha(self):
        self.assertAlmostEqual(mha([[1, 2, 3], [4, 5, 6]], {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}, 2), [0, 150], places=6)

    def test_transformer_block(self):
        self.assertAlmostEqual(transformer_block([[1, 2, 3], [4, 5, 6]], {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}, {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}, 2), [91, 233], places=6)

    def test_gpt2(self):
        self.assertAlmostEqual(gpt2([1, 2], {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}, 2), [91, 233], places=6)

    def test_generate(self):
        self.assertAlmostEqual(generate([1, 2], {"wte": [[1, 2, 3], [4, 5, 6]], "wpe": [[1, 2, 3], [4, 5, 6]], "blocks": [{"w": [[1, 2, 3], [4, 5, 6]], "b": 1}, {"w": [[1, 2, 3], [4, 5, 6]], "b": 1}], "ln_f": {"w": [[1, 2], [3, 4], [5, 6]], "b": 1}}, 2, 2), [1, 2], places=6)

    def test_main(self):
        self.assertIsInstance(main("Test"), str)

if __name__ == '__main__':
    unittest.main()