class DataAgent():
    def __init__(self,
                 task,
                 # dataset
                 train_shuffle = False,
                 train_batch_size: int = 8,
                 eval_batch_size: int = 20,
                 ) -> None:
        self.task = task
        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        
        self.eval_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=True)
        
        self.eval_data_iterator = self._infinite_data_loader(self.eval_dataloader)

        
    def get_actions(self, new_state_num = 2):
        return [next(self.train_data_iterator) for _ in range(new_state_num)]
    
    def get_train_batch(self):
        return next(self.train_data_iterator)
    
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
    
    def get_eval_batch(self): # a good place to add momentum
        return next(self.eval_data_iterator)