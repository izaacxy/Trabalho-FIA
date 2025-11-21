import os
import shutil
import sys

# 1. Remove a pasta local 'ltn' se ela existir (para não conflitar com a instalada)
if os.path.exists('ltn'):
    print("Pasta local 'ltn' encontrada. Removendo para usar a biblioteca oficial...")
    shutil.rmtree('ltn')
else:
    print("Nenhuma pasta local 'ltn' encontrada (isso é bom).")

# 2. Instala a biblioteca oficial (garantindo a versão correta)
!pip install git+https://github.com/logictensornetworks/LTNtorch

# 3. Hack para forçar o recarregamento sem reiniciar manual (mas reiniciar manual é melhor)
if 'ltn' in sys.modules:
    del sys.modules['ltn']
if 'ltn.core' in sys.modules:
    del sys.modules['ltn.core']

print("\n>>> PASSO CRÍTICO: Vá no menu acima 'Runtime' (Ambiente de Execução) -> 'Restart Session' (Reiniciar Sessão).")
print(">>> DEPOIS: Rode a segunda célula com o código do classificador.")
