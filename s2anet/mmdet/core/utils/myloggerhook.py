from mmcv.runner.hooks import Hook

class MyLoggerHook(Hook):


    def __init__(self, interval=1):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            secret_msg = runner.outputs.get("logger_images")
            print(secret_msg)