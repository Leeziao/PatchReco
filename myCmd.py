from subprocess import run, CompletedProcess
from pathlib import Path
import logging

class CMD:
	def __init__(self, cmd: str, path: Path=Path(), env=None) -> None:
		self._path = path.absolute()
		self._cmd = cmd
		self._env = env

	@property
	def env(self):
		return self._env
	@env.setter
	def env(self, value):
		self._env = value

	@property
	def path(self):
		return self._path
	@path.setter
	def path(self, value):
		self._path = value
	
	@property
	def cmd(self):
		return self._cmd
	@cmd.setter
	def cmd(self, value):
		self._cmd = value

	def __str__(self) -> str:
		return f"CMD[path: {self.path}\tcmd: {self.cmd}\tenv: {self.env}]"

	def __call__(self, shell: bool=True, getOutput: bool=False, check=True) -> CompletedProcess:
		logging.debug(self)
		r = run(self.cmd, 
			shell=shell, # run in a shell, otherwise cmd has to be a list of str
			check=check, # exception raise if non-zero return value
			capture_output=getOutput,
			cwd=self.path,
			env=self.env
		)
		return r

if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	c = CMD("ls -alh")
	r = c()
	c = CMD(["ls", "-alh"])
	r = c(shell=False)