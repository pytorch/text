from torchtext import datasets

# en-valid
TRAIN_NUM = [0] + [900] * 16 + [904, 905, 900, 904]
VAL_NUM = [0] + [100] * 16 + [96, 95, 100, 96]
TEST_NUM = [0] + [1000] * 20

# Testcase 1 (joint training)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, joint=True)
assert len(train_iter.dataset) == sum(TRAIN_NUM)
assert len(val_iter.dataset) == VAL_NUM[1]
assert len(test_iter.dataset) == TEST_NUM[1]

# Testcase 2 (only supporting)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, only_supporting=True)
assert len(train_iter.dataset) == TRAIN_NUM[2]
assert len(val_iter.dataset) == VAL_NUM[2]
assert len(test_iter.dataset) == TEST_NUM[2]

# Testcase 3 (single task)
for i in range(1, 21):
    train_iter, val_iter, test_iter = datasets.BABI20.iters(task=i)
    assert len(train_iter.dataset) == TRAIN_NUM[i]
    assert len(val_iter.dataset) == VAL_NUM[i]
    assert len(test_iter.dataset) == TEST_NUM[i]

# en-valid-10k
TRAIN_NUM = [0] + [9000] * 17 + [8996, 9000, 9002]
VAL_NUM = [0] + [1000] * 17 + [1004, 1000, 998]
TEST_NUM = [0] + [1000] * 20

# Testcase 1 (joint training)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, joint=True, tenK=True)
assert len(train_iter.dataset) == sum(TRAIN_NUM)
assert len(val_iter.dataset) == VAL_NUM[1]
assert len(test_iter.dataset) == TEST_NUM[1]

# Testcase 2 (only supporting)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, only_supporting=True,
                                                        tenK=True)
assert len(train_iter.dataset) == TRAIN_NUM[2]
assert len(val_iter.dataset) == VAL_NUM[2]
assert len(test_iter.dataset) == TEST_NUM[2]

# Testcase 3 (single task)
for i in range(1, 21):
    train_iter, val_iter, test_iter = datasets.BABI20.iters(task=i, tenK=True)
    assert len(train_iter.dataset) == TRAIN_NUM[i]
    assert len(val_iter.dataset) == VAL_NUM[i]
    assert len(test_iter.dataset) == TEST_NUM[i]
