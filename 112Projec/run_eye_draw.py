import os
import subprocess
import sys

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    print('Starting eye drawing prototype...')
    subprocess.Popen([sys.executable, 'main.py'], cwd=project_root)
