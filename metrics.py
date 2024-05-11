class Metrics:
    def __init__(self) -> None:
        super().__init__()
        self.__total_loss = 0.0
        self.__total_correct = 0.0
        self.__sample_num = 0.0
        self.__batch_num = 0
        #需要优化混淆矩阵

    def accuracy_epoch(self):
        return self.__total_correct / self.__sample_num

    def average_loss_epoch(self):
        return self.__total_loss / self.__batch_num

        # def precision(self):  # 不准错
        # def recall(self):  # 不准漏

    def add_in_epoch(self, sample_num, loss=0, correct=0):
        self.__total_loss += loss
        self.__total_correct += correct
        self.__sample_num += sample_num
        self.__batch_num += 1

    def reset(self):
        self.__total_loss = 0.0
        self.__total_correct = 0.0
        self.__sample_num = 0.0
        self.__batch_num = 0
