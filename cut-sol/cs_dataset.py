from torch.utils.data import Dataset
import csv


# Loads dataset for Cut Classifier - Testing
class CutClassTruths(Dataset):
    def __init__(self,folderPath):
        self.cut_truths = []
        self.decimal_cut_tt = []
        self.dist_cut_truths = set()
        with open(folderPath+"cuts.csv") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                self.dist_cut_truths.add(int(row[9]))
                self.dist_cut_truths.add(int(row[10]))

        for cut_tt in self.dist_cut_truths:  
            binary_string = bin(cut_tt)[2:].zfill(32)
            binary_list = [int(bit) for bit in binary_string]
            self.cut_truths.append(binary_list)
            self.decimal_cut_tt.append(cut_tt)

    def __getitem__(self, index):
        return self.cut_truths[index]
    
    def __len__(self):
        return len(self.cut_truths)
    

# Loads dataset for Delay Predictor - Testing
class CutQoRTruths(Dataset):
    def __init__(self,folderPath):
        self.cut_truth_map = {}
        self.cut_truths = []
        self.decimal_cut_tt = []
        self.dist_cut_truths = set()
        with open(folderPath+"cuts.csv") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                self.cut_truth_map[int(row[1])] = [int(row[9]), int(row[10])]

        with open(folderPath+"interim_cut_delays.csv") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if int(row[2]) == 1:
                    self.dist_cut_truths.add(self.cut_truth_map[int(row[0])][int(row[1])])

        for cut_tt in self.dist_cut_truths:
            binary_string = bin(cut_tt)[2:].zfill(32)
            binary_list = [int(bit) for bit in binary_string]
            self.cut_truths.append(binary_list)
            self.decimal_cut_tt.append(cut_tt)

    def __getitem__(self, index):
        return self.cut_truths[index]
    
    def __len__(self):
        return len(self.cut_truths)


# Loads dataset for Cut Classifier - Training
class CutClass(Dataset):
    def __init__(self,folderPath):
        self.truths = []
        self.class_labels = []
        dist_pairs = set()
        self.num_positive = 0

        with open(folderPath+"cut_stats.csv") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if row[3] == " -1.000000":
                    dist_pairs.add((int(row[2]),-1))
                else:
                    dist_pairs.add((int(row[2]),float(row[3])))

        for pair in dist_pairs:
            binary_string = bin(pair[0])[2:].zfill(32)
            binary_list = [int(bit) for bit in binary_string]
            self.truths.append(binary_list)
            if pair[1] == -1:
                self.class_labels.append(0)
            else:
                self.num_positive+=1
                self.class_labels.append(1)

    def __getitem__(self, index):
        return self.truths[index], self.class_labels[index]
    
    def __len__(self):
        return len(self.truths)
    

# Loads dataset for Delay Predictor - Training
class CutQoR(Dataset):
    def __init__(self,folderPath):
        self.truths = []
        self.reg_labels = []
        dist_pairs = set()

        with open(folderPath+"cut_stats.csv") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if row[3] != " -1.000000":
                    dist_pairs.add((int(row[2]),float(row[3])))

        for pair in dist_pairs:
            binary_string = bin(pair[0])[2:].zfill(32)
            binary_list = [int(bit) for bit in binary_string]
            self.truths.append(binary_list)
            self.reg_labels.append(pair[1])

    def __getitem__(self, index):
        return self.truths[index], self.reg_labels[index]
    
    def __len__(self):
        return len(self.truths)