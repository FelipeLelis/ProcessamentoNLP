

library(udpipe)

arquivo <- read.csv(file.choose(), sep=",", encoding = "UTF-8", header = F, quote="\"")

colnames(arquivo) = c("FILE","CLASS")
#1450 linhas
dim(arquivo)[1]
head(arquivo)

arquivo$FILE = as.character(arquivo$FILE)
arquivo$CLASS = as.character(arquivo$CLASS)

arquivo$ID = seq(1:dim(arquivo)[1])


modelo = udpipe_download_model(language = "english")

modelo = udpipe_load_model(file.choose())

#anotar 
anotar = udpipe_annotate(modelo, x = arquivo$FILE, doc_id = arquivo$ID)


anotar = as.data.frame(anotar)
head(anotar)
dim(anotar)


#filtra apenas verbos, substantetivos e adjetivos
anotar2 = anotar[anotar$upos %in% c("VERB", "NOUN", "ADJ","ADV"), ]
dim(anotar2)


#apenas colunas  doc_id, lemma
anotar2 = anotar2[,c(1,7)]

x <- document_term_frequencies(anotar2[, c("doc_id", "lemma")])
dtm <- document_term_matrix(x)
dim(dtm)
summary(dtm)
colnames(dtm)
rownames(dtm)
head(dtm)



df = as.data.frame(as.matrix(dtm))
#numero do doc esta no nome da linha, vamos passar para um coluna para fazer um merge com o documento original e pegar a classe
df$doc_id =   rownames(df)
df$doc_id = as.integer(df$doc_id)


df2 = merge(df,arquivo, by.x =c("doc_id") ,by.y = c("ID")  )
#REMOVEMOS A COLUNA COM ARQUIVO QUE FOI CRIADA NO MERGE
df2$FILE =  NULL
head(df2$CLASS,n=100)

dim(df2)



#install.packages("caret", dependencies=T)
library(caret)

#install.packages("FSelector", dependencies=T)
library(FSelector)

#selecao de atributos
atributos <- information.gain(CLASS ~., df2)
atributos2 = cutoff.k(atributos,20)
formula =  as.simple.formula(atributos2, class="CLASS")

#classe em fator
df2$CLASS = as.factor(df2$CLASS)

#particao
particao = createDataPartition(1:dim(df2)[1],p=.7)
dftreino = df2[particao$Resample1,]
dfteste = df2[- particao$Resample1,]
dim(dftreino)
dim(dfteste)


#executando random forest
library(randomForest)


modelo = randomForest(formula, data = dftreino, ntree=500 ) 
previsao  = predict(modelo, dfteste)
previsao
confusionMatrix(previsao, dfteste$CLASS)





