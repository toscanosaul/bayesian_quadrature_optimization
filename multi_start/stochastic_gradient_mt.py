

class StochasticGradient(object):

    def __init__(self, objective_function, gradient=None, training_data=None, batch_size=None):
        self.objective_function = objective_function
        self.training_data = training_data # it can be a training loader for NN
        self.gradient = gradient
        self.batch_size = batch_size


    def move_forward(self, current_point, epoch=1, batch_index=None, torch=True):
        value = 0.0
        if torch:
            value = self.move_forward_torch(current_point, batch_index, epoch)
        return value

    def move_forward_torch(self, batch_index, epoch, current_point=None):
        value = self.objective_function(batch_index, epoch)
        return value
