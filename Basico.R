
#install.packages("udpipe", dependencies = T)
library(udpipe)

#FILE = "Nossa vida é controlada por algoritmos,"
#Atualização para ficar com a codificação correta
FILE =  enc2utf8("Nossa vida é controlada por algoritmos,")

ID = 1L

arquivo = data.frame(FILE,ID)
arquivo$FILE = as.character(arquivo$FILE)




#download do modelo
modelo = udpipe_download_model(language = "portuguese-gsd")

#carregar o modelo do disco
modelo = udpipe_load_model(file.choose())

#anotacoes no fromato conll-u
anotar = udpipe_annotate(modelo, x = arquivo$FILE, doc_id = arquivo$ID)
anotar
anotar = as.data.frame(anotar)
colnames(anotar)

fix(anotar)


