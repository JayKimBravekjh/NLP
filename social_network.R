library(sqldf)
library(igraph)
library(RSQLite)
db <- dbConnect(dbDriver("SQLite"), "database.sqlite")
sn_data <- dbGetQuery(db, "SELECT p1.Name as Receiver, p2.Name as Sender from Persons p1, persons p2, emails e, EmailReceivers er where p1.Id = er.PersonId and p2.Id = e.SenderPersonId and e.id=er.EmailId")
sn_data_1 <- sqldf("select Receiver, Sender from sn_data group by Receiver, Sender having count(1) > 5 ")
gg <- graph.data.frame(as.matrix(sn_data_1))
gg_1 <- get.adjacency(gg, sparse=FALSE)
g <- graph.adjacency(gg_1, weighted=T, mode = "undirected")
g <- simplify(g)
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(299)
V(g)$label.cex <- 4*V(g)$degree / max(V(g)$degree) + .2
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$color <- degree(g) + 1
egam <- (log(E(g)$weight)+.4)/max(log(E(g)$weight)+.4)
E(g)$color <- rgb(.5, .5, 0, egam)
E(g)$width <- egam
# plot the graph in layout1
plot(g, layout = layout.kamada.kawai)








