import traceback
import torch 
import traceback

import utils


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    opts바탕하여 만든 Trainer

    Args:
        opt (:obj:`Namespace`): 사용자 옵션들
        model(:obj) : 트레이닝 할 모델
        fieids (dict) : 필드의 사전
        optim (obj: `utils.Optimizer`) : 트레이닝 할때에 쓰이는 옵티마이저
        data_type (str) : 데이터 타입을 알려주는 스트링문자
            e.g. "text
        model_saver(:obj : `models.ModelSaverBase`) : model을 저장을 위한 유틸리티 오브젝트
    """

    train_loss = utils.loss.build_loss_compute(model, fields, opt)
    valid_loss = utils.loss.build_loss_compute(model, fields, opt, train=False)

    trainer = Trainer(model, train_loss, valid_loss, optim)
    return trainer

class Trainer(object):
    """
    트레이닝 과정을 다루는 클래스
    Args:
        model(:obj) : 트레이닝 할 모델
        train_loss(:obj:`utils.loss.LossComputeBase`):
            training loss computation
        valid_loss(:obj:`utils.loss.LossComputeBase`):
            training loss computation
        optim (obj: `utils.Optimizer`) : 트레이닝 할때에 쓰이는 옵티마이저
        model_saver(:obj:`models.ModelSaverBase`): the saver is
                used to save a checkpoint.         
    """
    def __init__(self, model, train_loss, valid_loss, 
                    optim, model_saver=None ):
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.model_saver = model_saver

        # Set model in training mode.
        self.model.train()

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i],
                                          self.attention_dropout[i])
                # logger.info("Updated dropout/attn dropout to %f %f at step %d"
                #             % (self.dropout[i],
                #                self.attention_dropout[i], step))
    
    def _gradient_accumulation(self, true_batches):
    
        for k, batch in enumerate(true_batches):
            image, targets = batch
            self.optim.zero_grad()
            try:
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    outputs = self.model(image)
                    #compute loss
                    loss, batch_stats = self.train_loss(batch, outputs)
                step = self.optim.training_step

                if loss is not None:
                    self.optim.backward(loss)
            except Exception as exc:
                trace_content = traceback.format_exc()
                if "CUDA out of memory" in trace_content:
                    pass
                    # logger.info("Step %d, cuda OOM - batch removed",
                    #             self.optim.training_step)                    
                else:
                    traceback.print_exc()
                    raise exc







    def train(self,
            train_iter,
            train_steps,
            save_checkpoint_steps=5000,
            valid_iter=None,
            valid_steps=1000):

        """
        `train_iter`를 이터레이팅하는 메인 트레이닝 루프
        그리고 `valid_iter`를 통해 벨리데이션 할수도 있음!
        
        """

        if valid_iter is None:
            pass
        else:
            pass
      
        for i, batches in enumerate(train_iter):
            step = self.optim.training_step
            #Updata Dropout

            self._maybe_update_dropout(step)
            self._gradient_accumulation(
                batches
            )
