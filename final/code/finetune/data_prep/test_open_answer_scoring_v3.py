"""v3 판정 테스트. 실행: cd MMDL/virl39k/data_prep && python3 test_open_answer_scoring_v3.py
1) 기존 테스트 케이스는 v3에서도 같은 결과여야 함(의도적으로 바꾼 1건 제외)  2) 새 규칙 정답/오답 케이스."""
import sys; sys.path.insert(0, '.')
from open_answer_scoring_v3 import match_open, match_open_why, match_response
import test_open_answer_scoring as old   # 기존 테스트도 그대로 실행됨(출력 확인)

CHANGED = {('25%', '0.25')}   # v1은 오답, v3은 백분율 규칙으로 정답
bad = [(p, g, e) for p, g, e in old.T if (p, g) not in CHANGED and int(match_open(p, g)) != e]
print('v1 cases under v3', len(old.T) - len(bad), '/', len(old.T)); [print(' FAIL', b) for b in bad]

N = [  # (pred, gold, expected, 근거)
 ('$21.29 billion', '21.29', 1, 'number_words'), ('199.3 billion Qatari Riyal', '199.3', 1, 'number_words'),
 ('473,683 million yuan', '473683', 1, 'number_words'), ('1 minute per year', '1', 1, 'number_words'),
 ('4 grid squares', '5', 0, None), ('120 square meters', '96', 0, None), ('2 or 3 people', '2', 0, None),
 ('9 hearts', '11', 0, None), ('3 times', '3', 1, 'v1'),
 ('Yes, both halves will match perfectly.', 'Yes', 1, 'yes_no'), ('No, Dark Green does not have a higher value', 'No', 1, 'yes_no'),
 ('Yes, it is', 'No', 0, None), ('Not really', 'No', 0, None), ('**No**, because', 'no', 1, 'yes_no'),
 ('7.66%', '0.0766', 1, 'percent'), ('43%', '0.43', 1, 'percent'), ('8.86', '0.0886', 0, None), ('47%', '0.477', 0, None),
 ('0.43', '43%', 1, 'percent'), ('25%', '0.25', 1, 'percent'),
 ('(1, +\\infty)', 'x > 1', 1, 'interval'), ('[2, +∞)', 'x \\geq 2', 1, 'interval'), ('(-\\infty, 3)', 'x<3', 1, 'interval'),
 ('[1, +\\infty)', 'x > 1', 0, None), ('(1, +\\infty)', 'x < 1', 0, None),
 ('1', '4 - 3 = 1', 1, 'gold_equation'), ('2', '4 - 3 = 1', 0, None), ('5', 'x = 5', 1, 'v1'),
 ('2015', '2016', 0, 'int_fix'), ('2013', '2012', 0, 'int_fix'), ('2016', '2016', 1, 'v1'), ('3.141', '3.1416', 1, 'v1'),
 ('1,000', '1000', 1, 'v1'), ('1001', '1000', 0, 'int_fix'), (None, '5', 0, None)]
bad = [(p, g, e, w, match_open_why(p, g)) for p, g, e, w in N if match_open_why(p, g) != (bool(e), w)]
print('v3 new cases', len(N) - len(bad), '/', len(N)); [print(' FAIL', b) for b in bad]

R = [("So the total is **$21.29 billion**.", '21.29', 1), ("Final Answer: Yes, they match.", 'Yes', 1), ("Answer: 2015", '2016', 0)]
bad = [(r, g, e) for r, g, e in R if int(match_response(r, g)[0]) != e]
print('v3 match_response', len(R) - len(bad), '/', len(R)); [print(' FAIL', b) for b in bad]
