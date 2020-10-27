import solaris as sol
from os import makedirs
from os.path import dirname, join

src = dirname(__file__)
code = dirname(src)
yml = join(code, 'yml/')
tmp = join(code, 'tmp/')
model = join(tmp, 'model/')

training_data_csv = join(tmp, 'train.csv')
config_path = join(yml, 'train.yml')
model_checkpoint_path = join(model, 'checkpoint.pth')
model_dest_path = join(model, 'final.pth')

config = sol.utils.config.parse(config_path)

config['training_data_csv'] = training_data_csv
config['training']['callbacks']['model_checkpoint']['filepath'] = model_checkpoint_path
config['training']['model_dest_path'] = model_dest_path

trainer = sol.nets.train.Trainer(config=config)
trainer.train()
