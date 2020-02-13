###
###  PURPOSE:     Check data used in 2013
###               --> Biometrics 2016 paper
###
###  PROGRAMMER:  Arnošt Komárek
###
###  LOG:         20200213  created
###
### =======================================================
rm(list = ls())
options(width = 135)

ROOT <- "/home/komarek/teach/PhD_Kala/Tandmob/2013_intCensMisclass/"

(load(paste(ROOT, "/Data/Data_20130610.RData", sep = "")))

dim(Data2)
head(Data2)
  ## Standard interval-censored dataset which assumes no misclassification
  ## IDNR:       id dítěte
  ## TOOTH:      číslo zubu (zde pouze stoličky (šestky) ze 4 kvadrantů huby - 16, 26, 36, 46)
  ## EBEG, EEND: interval, ve kterém (poprvé) zaznamenáno prořezání zubu (může být zleva-cenzor., tj. EBEG = NA)
  ## FBEG, FEND: interval, ve kterém (poprvé) zaznamenán kaz (často zprava cenzorováno, tj. FEND = NA)
  ## GIRL:       pohlaví (1 = dívka, 0 = chlapec, další pohlaví se zde neuvažují)
  ## STARTBR:    věk (podle rodičů), kdy začal čistit zuby
  ## FREQ.BR:    0/1. It is equal to 1 if the child brushes daily the teeth, equal to 0 if he/she brushes less than once a day.
  ## SEAL:       0/1. It is equal to 1 if the permanent first molar xx was sealed in pits and fissures (a form of protection), 0 otherwise.
  ## PLAQUE.1:   0/1. It is equal to 1 if there was occlusal plaque in pits and fissures of the tooth. It is equal to 0 if there was either no plaque present or the plaque was present on the total occlusal surface.
  ## PLAQUE.2:   0/1. It is equal to 1 if there was occlusal plaque on the total occlusal surface of the tooth. It is equal to 0 if there was either no plaque present or the plaque was present only in pits and fissures.
  ## MEALS:      0/1. It is equal to 1 if the number of between-meal snacks is more than two a day. It is equal to 0 if the number of between-meal snacks is two or less a day.
  ## XCEN:       standardized x coordinate of the municipality of the school to which the child belongs.
  ## YCEN:       standardized y coordinate of the municipality of the school to which the child belongs.
  ## FTimeImp:   mnou vyrobený init pro čas kazu (prostředek intervalu, resp. dolní mez, pokud zprava cenzorováno), používáno k výpočtů smysluplných init hodnot pro MCMC apod.

length(unique(Data2[, "IDNR"]))  ## 3896 children
sum(is.na(Data2[, "EBEG"]))      ## 0
sum(is.na(Data2[, "FEND"]))      ## 11494
sum(!is.na(Data2[, "FEND"]))     ## 3979

dim(Data1)
head(Data1)

## IdTooth:  zkombinované id dítě-zub
## VISIT:    the age at each visit, in years.
## EXAMINER: the examiner who examines the child at each examination,
##           - celkem 16 hodnotitelů - 1, 2, ..., 16
## STATUS:   variable 0/1 indicating whether the tooth had caries or not at each examination
##           - potenciálně misklasifikováno
## TOOTH:    variable indicating whether the information corresponds to tooth 16, 26, 36, 46.

table(Data1[, "EXAMINER"])
