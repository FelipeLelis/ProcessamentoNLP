

library(udpipe)

arquivo <- read.csv(file.choose(), sep=";", encoding = "UTF-8")
arquivo$ID = as.integer(arquivo$ID)
arquivo$FILE = as.character(arquivo$FILE)

dim(arquivo)
colnames(arquivo)

modelo = udpipe_load_model(file.choose())


#anotacoes no fromato conll-u
anotar = udpipe_annotate(modelo, x = arquivo$FILE, doc_id = arquivo$ID)
anotar = as.data.frame(anotar)
fix(anotar)
dim(anotar)


anotar$upos



#sumariza frequencia upos
frequencia = txt_freq(anotar$upos)
barplot(frequencia$freq, col = gray.colors(14),xlab="Tipo",ylab="Frequência", names.arg = frequencia$key)


#palavras chave - combinação
anotar$palavra = tolower(anotar$token)
estatisticas = keywords_collocation(x = anotar, term="token", group = "doc_id")
estatisticas$chave = factor(estatisticas$keyword, levels= rev(estatisticas$keyword))
estatisticas2 = head(subset(estatisticas,freq>2))
barplot(estatisticas2$pmi, col = gray.colors(14),xlab="Tipo",ylab="Frequência", names.arg =estatisticas2$chave )






