import sys; sys.path.insert(0, '.')
from open_answer_scoring import extract_open, match_open
T = [  # (pred, gold, expected)
 ('12π','12\\pi',1),('12 \\pi','12\\pi',1),('37.7','12\\pi',1),('37','12\\pi',0),
 ('40/3','\\frac{40}{3}',1),('13.33','\\frac{40}{3}',1),('13','\\frac{40}{3}',0),
 ('130°','130^{\\circ}',1),('130 degrees','130^{\\circ}',1),('120°','130^{\\circ}',0),
 ('(1, 1, 1)','(1,1,1)',1),('(1,1,2)','(1,1,1)',0),('[-4, 0)','[-4,0)',1),('(-4,0)','[-4,0)',0),
 ('10','10 cm',1),('10 cm','10cm',1),('100','10 cm',0),('No','No',1),('no.','No',1),('Yes','No',0),
 ('1024','{1024}',1),('2πR^3','2\\pi R^3',1),('2\\pi R^{3}','2\\pi R^3',1),('2πR^2','2\\pi R^3',0),
 ('4n + 2','(4n+2)',1),('3n+1','3n+1',1),('3n+2','3n+1',0),('reebok','Reebok',1),('Nike','Reebok',0),
 ('2023/24','2023/24',1),('30','30 and 0.25',0),('30 and 0.25','30 and 0.25',1),('62°','62°',1),('x = 5','5',1),
 ('-7','-7',1),('7','-7',0),('0.25','1/4',1),('25%','0.25',0),('3.2 m','3.20 m',1),('\\sqrt{3}','√{3}',1),('90° - 2x°','90°-2x°',1),
 (None,'5',0),('','5',0)]
bad = [(p, g, e, match_open(p, g)) for p, g, e in T if int(match_open(p, g)) != e]
print('match_open', len(T) - len(bad), '/', len(T)); [print(' FAIL', b) for b in bad]
E = [("Reasoning...\n\n**Answer: 12π**",'12π'),("so \\boxed{\\frac{40}{3}} done",'\\frac{40}{3}'),("Thus the answer is 17.",'17'),
     ("✅ Final Answer: **D**",'D'),("**Final Answer:** 3.2 m",'3.2 m'),("blah\nAnswer: (1, 1, 1)\n",'(1, 1, 1)'),
     ("The final answer is **No**.",'No'),("no answer here at all",None),("first \\boxed{2} then \\boxed{5}",'5')]
bad = [(r, w, extract_open(r)) for r, w in E if extract_open(r) != w]
print('extract_open', len(E) - len(bad), '/', len(E)); [print(' FAIL', b) for b in bad]

from open_answer_scoring import match_response
R=[("...so\n\n✅ So, the process is: **Transformation**.",'Transformation',1),
   ("blah\n\n**0.015625** or **1/64**\n\n---\n\n✅ So, the probability is **1/64**.",'1/64',1),
   ("Final Answer:\n\\[\n\\boxed{13.0 \\, \\text{g}}\n\\]",'13.0 g',1),
   ("work work\n0.0630",'0.063',1),("work\n\nSo the answer is **2**.",'5',0),
   ("long reasoning with the number 5 and 7 in the middle but no final answer at all, it just rambles on",'5',0),
   ("Answer: > **trans-1-chloro-4-methylcyclohexane**",'trans-1-Chloro-4-methylcyclohexane',1),
   ("So answer is **12π**.",'12\\pi',1)]
bad=[(r,g,e,match_response(r,g)[0]) for r,g,e in R if int(match_response(r,g)[0])!=e]
print('match_response',len(R)-len(bad),'/',len(R)); [print(' FAIL',b) for b in bad]
