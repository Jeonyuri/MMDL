"""grpo_reward.py 단위 테스트. 실행: python3 test_grpo_reward.py (math_verify 필요)"""
from grpo_reward import score

CH = ['red', 'blue', 'green', 'yellow']


def r(text, kind, gold, choices=(), finish='stop'):
    return score(text, finish, kind, gold, list(choices))['reward']


def test_mc_matches_v2():
    assert r('Answer: B', 'mc', 'B', CH) == 1
    assert r('The answer is B.', 'mc', 'B', CH) == 1
    assert r('Therefore, the correct answer is **B**.', 'mc', 'B', CH) == 1      # 기존 GRPO 노트북 보상은 0
    assert r('### Final Answer: **B. blue** ✅', 'mc', 'B', CH) == 1             # 기존 GRPO 노트북 보상은 0
    assert r('> **B. blue**', 'mc', 'B', CH) == 1
    assert r('Answer: C', 'mc', 'B', CH) == 0
    assert r('I am not sure.', 'mc', 'B', CH) == 0


def test_truncated_is_zero():
    assert r('Answer: B', 'mc', 'B', CH, finish='length') == 0
    assert r('Answer: 12', 'open', '12', finish='length') == 0


def test_open_v3():
    assert r('Answer: 12', 'open', '12') == 1
    assert r('Answer: 28 meters', 'open', '28') == 1         # v3 number_words (기존 GRPO 노트북 보상은 0)
    assert r('Answer: 2016', 'open', '2015') == 0            # v1 정수 허용오차 버그 제거
    assert r('Answer: 50%', 'open', '0.5') == 1              # v3 percent
    assert r('Yes, the segment is larger.', 'open', 'Yes') == 1


def test_or_guard_blocks_hedging():
    assert r('Answer: **3** or **5**', 'open', '3') == 0
    assert r('Answer: **5** or **3**', 'open', '3') == 0
    assert r('Answer: **0.5** or **50%**', 'open', '0.5') == 1   # 같은 값의 다른 표기는 통과
    assert r('The answer is more or less clear: **7**', 'open', '7') == 1


def test_mc_conflict_mask():
    s = score('Option C says blue.\n\nFinal Answer: **B**', 'stop', 'mc', 'B', CH)
    assert isinstance(s['mask'], bool)
    s = score('Answer: B', 'stop', 'mc', 'B', CH)
    assert s['reward'] == 1 and not s['mask']


def test_gold_self_check():
    from grpo_reward import gold_self_check
    assert gold_self_check('mc', 'B', CH)
    assert gold_self_check('open', '12', [])
    assert not gold_self_check('open', '>', [])      # 알려진 한계: '>' 정답은 채점 불가 → 문항 제외


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
