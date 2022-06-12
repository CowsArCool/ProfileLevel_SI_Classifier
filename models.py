# Torch
import torch
import torch.nn as nn
import torch.optim as optim
# Using standard huggingface tokenizer for compatability
from transformers import BertModel, get_linear_schedule_with_warmup

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc, recall, precision


class SuicideClassifier (pl.LightningModule):
    def __init__(
            self, output_classes: list = ['suicide'],
            training_steps: int = None,
            warmup_steps: int = 0, lr=None,
            metrics=[]):

        super().__init__()

        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.output_classes = output_classes
        self.output_dim = len(output_classes)

        self.bert = BertModel.from_pretrained(BERT_MODEL, return_dict=True)
        self.ff = nn.Linear(self.bert.config.hidden_size, self.output_dim)
        self.output_norm = nn.Sigmoid()

        # loss loss function
        self.criterion = nn.BCELoss()
        self.lr = lr

        self.metrics = metrics
        self.implemented_metrics = {
            'ROC': self.calculate_ROC,
            'binary_report': self.calculate_binary_report
        }

        self.saved_metric_scores = {
            key: {
                metric: 0
                for metric in ['accuracy', 'f1 score', 'precision', 'recall_count']
            } for key in ['train', 'valid', 'test']
        }

    def forward(self, input_ids, attention_mask,
                labels=None, normalize=True):
        """ Preforms a forward pass through the model 
            and runs loss calculations

        Args:
            input_ids (torch.tensor[N, max_example_len]): integer incoded words
            attention_mask (torch.tensor[N, max_example_len]): mask for self attention (1:unmasked, 0: masked)
            labels (torch.tensor [N]): ground truth y values for batch
        """
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)

        # with return_dict=True, bert outputs
        x = self.bert(input_ids, attention_mask=attention_mask)
        y_hat = self.ff(x.pooler_output)
        if normalize:
            y_hat = self.output_norm(y_hat)

        if self.output_dim == 1:
            y_hat = torch.squeeze(y_hat)

        loss = 0
        if labels is not None:
            # print (f'y_hat type {(y_hat.dtype)}, labels type {(labels.dtype)}')
            loss = self.criterion(y_hat, labels.type(torch.float32))

        return loss, y_hat

    def _step(self, batch, step_type):
        (input_ids, attention_mask), labels = batch
        loss, output = self(input_ids, attention_mask, labels)
        self.log('{}_loss'.format(step_type), loss, prog_bar=True, logger=True)

        return {f'loss': loss, f'output': output, f'labels': labels}

    def training_step(self, batch, batch_idx):
        values = self._step(batch, 'train')
        return values

    def validation_step(self, batch, batch_idx):
        values = self._step(batch, 'valid')
        return values

    def test_step(self, batch, batch_idx):
        values = self._step(batch, 'test')
        return values['loss']

    def calculate_ROC(self, preds, labels, step_type):
        for i, name in enumerate(self.output_classes):
            if self.output_dim == 1:
                # class_roc_auc = auroc (preds, labels)
                i = None

            class_roc_auc = auroc(preds[:, i], labels[:, i], pos_label=1)

            self.log(
                f"{name}_roc_auc/{step_type}", class_roc_auc, self.current_epoch
            )

    def calculate_binary_report(self, preds, labels, step_type):
        assert len(
            labels.shape) == 1, 'binary report is reserved for output_dim==1'
        assert len(preds.shape) == 1

        # print (f'shapes| preds: {preds.shape} labels: {labels.shape} types| preds: {preds.dtype} labels: {labels.dtype}')

        binary_metrics = {
            'accuracy': [accuracy],
            'f1 score': [f1, {'num_classes': 1}],
            'precision': [precision],
            'recall_count': [recall]
        }

        for name, metric_info, in binary_metrics.items():
            kwargs = {}
            if len(metric_info) > 1:
                kwargs = metric_info[1]

            metric_score = metric_info[0](preds, labels, **kwargs)
            self.log('{}/{}'.format(name, step_type), metric_score)

            if self.saved_metric_scores[step_type][name] < metric_score:
                self.saved_metric_scores[step_type][name] = metric_score

            self.log('max {}/{}'.format(name, step_type),
                     self.saved_metric_scores[step_type][name])

        # self.log_dict(self.saved_metric_scores[step_type])

    def log_metrics(self, outputs, step_type):
        labels, preds = [], []

        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["output"].detach().cpu():
                preds.append(out_predictions)

        labels = torch.stack(labels).int()
        preds = torch.stack(preds)

        for metric in self.metrics:
            self.implemented_metrics[metric](preds, labels, step_type)

    def log_min_loss(self, outputs):
        loss_vector = []
        print(f'outputs len: {len(outputs)}')
        for output in outputs:
            print(f'output len: {len(output)}')
            print(f'output loss len: {len(output["loss"])}')
            for loss in output['loss'].detach().cpu():
                loss_vector.append(loss)

        min_loss = torch.mean(loss_vector).item()
        if min_loss < self.min_valid_loss:
            self.min_valid_loss = min_loss
        self.log('min_loss/valid', self.min_valid_loss)

    def training_epoch_end(self, outputs):
        self.log_metrics(outputs, 'train')

        return

    def validation_epoch_end(self, outputs):
        self.log_metrics(outputs, 'valid')
        # self.log_min_loss(outputs)

        return

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


BERT_MODEL = 'bert-base-uncased'
