class RLModel(BaseModel):

    def __init__(self,
                 model,
                 reward,
                 criterion,
                 critic,
                 optimizer,
                 scheduler,
                 beam_width: int,):
        
        super().__init__(model, criterion, optimizer, scheduler)

        self.reward = reward
        self.critic = critic
        
    def training_step(self, batch, batch_idx):
        # Perform sampling, (B,T), (B,T)
        gen_result, sample_logprobs = self.model(img_feats, img_masks, boxes, opt = {'sample_max':0}, mode = 'sample')
        # Get self critical reward, (B,T)
        reward = self.reward(self.model, img_feats, img_masks, boxes, data['gts'], gen_result)
        # Compute loss, ([])
        loss = self.critic(sample_logprobs, gen_result.data, reward)
        self.log('train/loss', loss, on_step = True, on_epoch = False, 
                 logger = True, prog_bar = True)
        return loss