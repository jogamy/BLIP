import os

class BaseData():
    def __init__(self, inf_DIR, ans_DIR) -> None:
        self.inf_seqs, self.inf_ips = [], []
        self.inps, self.ans_seqs, self.ans_ips = [], [], []

    def load_inf(self, inf_DIR):
        with open(os.path.join(inf_DIR, 'seq.txt'), 'r', encoding='utf-8') as f:
            seqs = [line.strip() for line in f]
        with open(os.path.join(inf_DIR, 'ip.txt'), 'r', encoding='utf-8') as f:
            ips = [line.strip() for line in f]
        
        return seqs, ips
        

class BaseLog():
    def __init__(self) -> None:
        pass


class BaseEvaluator():
    def __init__(self) -> None:
        pass

    @staticmethod
    def em_seq(infs, anss, trigger_func):
        correct = 0
        err_list =[]
        for index, (inf, ans) in enumerate(zip(infs, anss)):
            if inf == ans:
                correct += 1
            else:
                if trigger_func(inf == ans):
                    err_list.append(index)
        
        print(correct / len(anss))
        
        return err_list
        

