from copy import deepcopy
import os
import torch
from collections import deque

class ModelSaverBase(object):
    """
    모델 세이빙의 베이스 클래스.
    상속받는 클래스들은 반드시 프라이빗 메소드로 다음을 실행해야함:
    * `_save`
    * `_rm_checkpoint`
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=-1) -> None:
        self.base_path = base_path
        self.model  = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_save_step = None
        self.keep_checkpoint = keep_checkpoint

        if keep_checkpoint > 0:
            self.checkpotnt_queue = deque([], maxlen = keep_checkpoint)

    def save(self, step, moving_average=None):
        """
        Main entry point for model saver
        """

        if self.keep_checkpoint == 0 or step == self.last_save_step:
            return
        
        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data
        
        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step, model):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            model (nn.Module): torch model to save

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()

class ModelSaver(ModelSaverBase):

    def _save(self, step, model):
        model_state_dict = model.state_dict()
        model_state_dict = {k:v for k, v in model_state_dict.items() 
                            if 'generator' not in k}
        generator_state_dict = model.generator.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        # logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path


    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)