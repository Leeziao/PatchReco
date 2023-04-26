from myCmd import CMD
import multiprocessing as mp

def dispatch(modelType: str):
    CMD(f'python main.py {modelType}')()

if __name__ == '__main__':
	pool = mp.Pool(2)
	pool.map(dispatch, ['msg', 'code'])

	print('FINISH!')
