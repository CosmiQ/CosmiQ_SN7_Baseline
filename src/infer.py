import solaris as sol
from os import makedirs
from os.path import dirname, join

src = dirname(__file__)
code = dirname(src)
yml = join(code, 'yml/')
tmp = join(code, 'tmp/')
model = join(tmp, 'model/')

inference_data_csv = join(tmp, 'infer.csv')
config_path = join(yml, 'infer.yml')
model_dest_path = join(model, 'final.pth')
output_dir = join(tmp, 'raw')

config = sol.utils.config.parse(config_path)

config['model_path'] = model_dest_path
config['inference_data_csv'] = inference_data_csv
config['inference']['output_dir'] = output_dir

inferer = sol.nets.infer.Inferer(config)
inferer()
