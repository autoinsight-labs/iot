# AutoInsight

## 🚀 Sobre o Projeto

O **AutoInsight Plate Detector** é um sistema de visão computacional desenvolvido em Python para detecção e reconhecimento automático de placas de motocicletas brasileiras. Utilizando YOLOv11 para detecção de veículos e EasyOCR para reconhecimento de texto, o sistema oferece alta precisão na identificação de placas no formato Mercosul (ABC1D23) com processamento batch e geração automática de recortes.

## 👥 Equipe de Desenvolvimento

| Nome | RM | E-mail | GitHub | LinkedIn |
|------|-------|---------|---------|----------|
| Arthur Vieira Mariano | RM554742 | arthvm@proton.me | [@arthvm](https://github.com/arthvm) | [arthvm](https://linkedin.com/in/arthvm/) |
| Guilherme Henrique Maggiorini | RM554745 | guimaggiorini@gmail.com | [@guimaggiorini](https://github.com/guimaggiorini) | [guimaggiorini](https://linkedin.com/in/guimaggiorini/) |
| Ian Rossato Braga | RM554989 | ian007953@gmail.com | [@iannrb](https://github.com/iannrb) | [ianrossato](https://linkedin.com/in/ianrossato/) |

## 🛠️ Tecnologias Utilizadas

### Stack Principal
- **Python 3.8+** - Linguagem principal
- **YOLOv11** - Detecção de objetos (veículos)
- **EasyOCR** - Reconhecimento óptico de caracteres
- **OpenCV** - Processamento de imagens
- **NumPy** - Operações numéricas

### Bibliotecas Especializadas
- **Ultralytics** - Framework YOLOv11
- **Re (Regex)** - Validação de padrões de placas
- **Argparse** - Interface de linha de comando
- **Glob** - Busca de arquivos em lote

### Formatos Suportados
- **Imagens**: JPEG, PNG, BMP, TIFF
- **Placas**: Formato Mercosul (ABC1D23)
- **Saída**: Recortes anotados em JPEG

## 🎯 Funcionalidades Principais

### ✅ Detecção Inteligente
- **Detecção multi-escala** com YOLOv11x para máxima precisão
- **Fallback adaptativo** para diferentes condições de imagem
- **Processamento de regiões** quando veículos não são detectados
- **Eliminação de duplicatas** em detecções sobrepostas

### ✅ Reconhecimento Avançado
- **OCR otimizado** para português brasileiro
- **7 técnicas de pré-processamento** de imagem
- **Correção automática** de erros comuns de OCR
- **Validação rigorosa** do formato Mercosul

### ✅ Processamento Robusto
- **Processamento em lote** de diretórios inteiros
- **Ordenação natural** de arquivos numerados
- **Tratamento de erros** com continuidade de processamento
- **Múltiplas tentativas** com diferentes configurações

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/autoinsight-labs/iot.git
   cd iot
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o sistema:**
   ```bash
   # Imagem única
   python main.py -i images/image_1.jpg
   
   # Diretório completo
   python main.py -i ./images/
   ```

## 📋 Comandos e Parâmetros

### Sintaxe Básica
```bash
python main.py --input <caminho> [--model <modelo>]
```

### Parâmetros Disponíveis
| Parâmetro | Abreviação | Descrição | Padrão | Obrigatório |
|-----------|------------|-----------|---------|-------------|
| `--input` | `-i` | Caminho para imagem ou diretório | - | ✅ |
| `--model` | `-m` | Modelo YOLO a utilizar | `yolo11x.pt` | ❌ |

## 💡 Exemplos de Uso

### Processamento de Imagem Única
```bash
python main.py -i images/image_1.jpg
```


### Processamento em Lote
```bash
# Processar todas as imagens de um diretório
python main.py -i ./images/
```

### Saída Esperada

#### Imagem Única
```
Detected plates:
- ABC1D23

Plate crops saved to: output/
```

#### Exemplo de imagem gerada
![image](https://github.com/user-attachments/assets/ea1f01a5-5c17-4ae4-8747-131ffb8e166f)


#### Processamento em Lote
```
Processing 1/15: moto_001.jpg
Processing 2/15: moto_002.jpg
...
Processing 15/15: moto_015.jpg

Directory Processing Complete:
- Total images processed: 15
- Images with plates detected: 12
- Total plates found: 14
- Plate crops saved to: output/

Detailed Results:
📁 moto_001.jpg
  - ABC1D23
📁 moto_002.jpg
  - DEF2G45
📁 moto_003.jpg - No plates detected
```

## 📊 Especificações Técnicas

### Formatos de Placa Suportados
- **Mercosul**: ABC1D23 (3 letras + número + letra + 2 números)
- **Validação flexível** para casos com OCR imperfeito
- **Correção automática** de confusões comuns (I↔1, O↔0, etc.)

### Performance
- **Tempo médio**: 1-3 segundos por imagem
- **Precisão**: >75% em condições normais
- **Throughput**: ~20-60 imagens/minuto (dependendo do modelo)

### Requisitos de Sistema
- **RAM**: Mínimo 4GB, recomendado 8GB+
- **Armazenamento**: ~200MB para dependências
- **GPU**: Opcional, acelera processamento significativamente

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do challenge da Mottu FIAP.
