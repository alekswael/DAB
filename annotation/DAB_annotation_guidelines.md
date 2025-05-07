# Guide til annotering af tekstdokumenter

## Oversigt
Dette dokument indeholder retningslinjer for annotering af tekst materiale, således at det bliver anonymiseret mht. GDPR lovgivningen. Du får en række tekstdokumenter, hvori der optræder en eller flere personer, hvis identitet skal beskyttes. Målet med din annotering er at vurdere, hvilke dele af teksten, som skal maskeres, så teksten bliver anonymiseret. _Læs venligst instruktionerne nedenfor grundigt, inden du går i gang med at annotere._

For hvert dokument består din opgave af 4 trin, som vil hjælpe dig igennem processen:

- __Trin 0: Gennemlæsning.__ Læs hele dokumentet igennem.
- __Trin 1: Tekstenheder.__ I trin 1 skal du annotere alle _tekstenheder_ du kan finde, som svarer til en af 8 _semantiske kategorier_ (kan ses i tabellen under trin 1). 
- __Trin 2: Maskering.__ For hver tekstenhed, som du har annoteret i trin 1, skal du nu angive hvorvidt den skal maskeres, da den kan bruges til at direkte- eller indirekte identificere en person (_identifikator-type_), samt hvorvidt tekstenheden falder under en af 5 _fortrolige kategorier_ (kan ses i tabellerne under trin 2).
- __Trin 3: Sidste gennemgang.__ Når du er færdig med din annotering skal du til sidst gennemgå dokumentet for at sikre, at du ikke har overset nogle tekstenheder. Såfremt du opdager fejl/mangler, skal du korrigere dem ved at gennemgå trin 1 og 2 igen.

## Trin 1: Tekstenheder
I trin 1 skal du finde og markere tekstenheder. En tekstenhed er en strækning af et eller flere ord og tegn, som tilsammen udgør en enhed, der passer i en semantisk kategori. Listen over semantiske kategorier præsenteres i tabellen nedenfor. Hvis tekstenheden ikke passer ind i en af disse kategorier, skal MISC-kategorien bruges (se _Eksempler på semantiske kategorier_).

<table border="1">
  <tr>
    <th bgcolor="#dddddd">Semantisk kategori</th>
    <th bgcolor="#dddddd">Beskrivelse</th>
  </tr>
  <tr>
    <td>PERSON</td>
    <td>Navne på personer, herunder aliaser/kaldenavne, brugernavne og initialer</td>
  </tr>
  <tr>
    <td>CODE</td>
    <td>Omfatter ID-numre og koder, der identificerer noget, som SSN, telefonnummer, pasnummer,bilnummerplade osv.</td>
  </tr>
  <tr>
    <td>LOC</td>
    <td>Omfatter byer, områder, lande, adresser samt navngivne infrastrukturer som busstoppesteder, broer osv.</td>
  </tr>
  <tr>
    <td>ORG</td>
    <td>Omfatter navne på organisationer, virksomheder, offentlige institutioner, skoler, NGO'er osv.</td>
  </tr>
<tr>
    <td>DEM</td>
    <td>Demografiske egenskaber som alder, etnicitet, jobtitler, uddannelse, fysiske beskrivelser.</td>
  </tr>
  <tr>
    <td>DATETIME</td>
    <td>Specifikke datoer, tidspunkter eller varigheder.</td>
  </tr>
  <tr>
    <td>QUANTITY</td>
    <td>Betydningsfulde mængder, som procentdele eller monetære værdier.</td>
  </tr>
  <tr>
    <td>MISC</td>
    <td>Alle andre oplysninger, der beskriver en person, men ikke falder ind under de andre kategorier.</td>
  </tr>
</table>

For at gøre din opgave nemmere er dokumenterne blevet præ-annoteret - det vil sige, at nogle tekstenheder allerede er blevet markeret (dog er der _ikke_ angivet identifikator-type).

__OBS: _Præ-annoteringerne er fejlbarlige, og er blot et udgangspunkt for din annotering - du skal aktivt rette i annoteringerne samt tilføje dine egne.___ 

Sådan gør du:

1. __Kontroller præ-annoteret tekstenheder.__ Tekstenheder er markeret med en farve og kode, som viser hvilken semantisk kategori de tilhører. Klik på en tekstenhed for at ændre den. Kontroller, om hver præ-annoteret tekstenhed er annoteret korrekt, eller skal fjernes/redigeres, eksempelvis fordi den semantiske kategori er forkert eller ordstrækningen skal ændres.
2. __Annoter nye tekstenheder.__ Annoter herefter nye tekstenheder, der ikke er blevet opdaget af det automatiske værktøj. En tekstenhed skal passe i en af de 8 semantiske kategorier. Vælg først en kategori ved at klikke på den i venstre side af displayet, eller brug tal-knapperne på tastaturet til hurtigt at vælge kategori. Marker herefter tekstenheden med musen, og tekstenheden markeres med den farve og kode, som hører til kategorien.

Når du annoterer, skal du generelt markere den _mindste ordstrækning_, der angiver den pågældende tekstenhed. Dette betyder eksempelvis, at et mellemrum efter det sidste ord i ordstrækningen ikke skal inkluderes i tekstenheden.

I tilfælde af, at to eller flere tekstenheder refererer til det samme underliggende tekstenhed (f.eks. "Hans Jørgensen" og "Hr. Jørgensen"), skal du oprette en relation mellem tekstenhederne (se mere i afsnittet _Relationer_ under _Eksempler på kategorier_).

__OBS:__ I trin 1 behøver du ikke bekymre dig om hvorvidt tekstenhederne kan bruges til re-identifikation af personer (det kommer i Trin 2), du skal blot annotere <u>alle</u> tekstenheder, der tilhører en af de semantiske kategorier.
### Eksempler på semantiske kategorier
#### PERSON
For personnavne skal annoteringen inkludere titler og tiltaleformer, såsom Hr., Dr., osv., da disse kan bidrage til at identificere en person.
- <u>Hr. Gestur Jónsson</u><sub>(PERSON)</sub> og <u>Hr. Ragnar Halldór Hall </u><sub>(PERSON)</sub>
- <u>Hr. og Fru Jørgensen</u><sub>(PERSON)</sub>

Eksempler på, hvad der betragtes som navne:
- __Navne:__ F.eks. 'Hans Jørgensen, 'Jørgensen', 'Hans'
- __Initialer:__ F.eks. 'H.H.'
- __Stavefejl:__ F.eks. 'Hans Jørnsen'
- __Alle ortografiske variationer:__ F.eks. 'hans jørgensen', 'Hans JØRGENSEN'
#### CODE
Omfatter ID-numre, koder og andre talrækker, der identificerer noget, såsom CPR-nummer, telefonnummer, pasnummer, nummerplade, rapportnumre osv.
- Angående sagsnummer (nr. <u>42552/98</u>)<sub>(CODE)</sub>
#### LOC
Omfatter byer, områder, kommuner, adresser samt andre geografiske steder, bygninger og faciliteter. Andre eksempler er lufthavne, kirker, restauranter, hoteller, turistattraktioner, hospitaler, butikker, adresser, veje, have, fjorde, bjerge og parker.
- <u>Reykjavik</u><sub>(LOC)</sub>
- <u>Øvregaten 2a, 5003 Bergen</u><sub>(LOC)</sub>

Inkludér tal, når de er en del af navnet:
- <u>Pilestredet 48</u><sub>(LOC)</sub>
- <u>Rema 1000</u><sub>(ORG)</sub>

Annoter altid hele ordstrækningen, selv hvis flere ord indgår i samme tekstenhed:
- <u>Høgskolen i Oslo og Akershus</u><sub>(ORG)</sub>

Hvis der er tale om separate tekstenheder forbundet med en konjunktion (f.eks. "og"), skal de annoteres separat:
- Gågaderne i <u>Viborg</u> og <u>Silkeborg</u><sub>(LOC)</sub>
#### ORG
Omfatter enhver navngiven samling af mennesker, virksomheder, institutioner, organisationer, universiteter, hospitaler, kirker, sportshold, fagforeninger, politiske partier osv.

Firmaangivelser som AS, Co. og Ltd. skal inkluderes som en del af navnet.
- <u>A.P. Møller - Mærsk A/S</u><sub>(ORG)</sub>

Oversættelser og akronymer inkluderes i markeringen, f.eks:
- <u>Aarhus Universitet (AU)</u><sub>(ORG)</sub>

Bestemte eller ubestemte artikler medtages typisk ikke i markeringen, medmindre de eksplicit er en del af tekstenheden.
- <u>Det Danske Sprog- og Litteraturselskab</u><sub>(ORG)</sub>
- Sidste års vinder af <u>Den Store Bagedyst</u><sub>(MISC)</sub>

Hvis en tekstenhed kan tilhører flere kategorier, skal du vælge den semantiske kategori, der bedst beskriver enheden ud fra konteksten:
- <u>Sverige</u><sub>(ORG)</sub> har opkøbt en strategisk Bitcoin reserve
- <u>Sverige</u><sub>(LOC)</sub> ligger øst for <u>Norge</u><sub>(LOC)</sub>
#### DEM
Disse er demografiske markører, og omfatter både fysiske, kulturelle og erhvervsmæssige/uddannelsesmæssige attributter, såsom fysiske beskrivelser, diagnoser, modersmål, etnicitet, jobtitler, alder osv.
- <u>40 år</u><sub>(DEM)</sub> gammel
- ansøgeren er <u>journalist</u><sub>(DEM)</sub>
- en gruppe <u>venstreorienterede</u><sub>(DEM)</sub> ekstremister
- diagnosticeret med <u>brystkræft</u><sub>(DEM)</sub>
- en <u>svensk</u><sub>(DEM)</sub> fysiker

Pronominer (han, hun) skal ikke medtages i tekstenhederne.
#### DATETIME
Præpositioner (f.eks. på, ved) skal ikke inkluderes i markeringen.
- <u>Mandag, 3. oktober 2018</u><sub>(DATETIME)</sub>
- kl. <u>9:48</u><sub>(DATETIME)</sub>
- født i <u>1947</u><sub>(DATETIME)</sub>

Separate tekstenheder forbundet med "og" skal annoteres separat:
- <u>10. marts</u><sub>(DATETIME)</sub> og <u>12. marts</u><sub>(DATETIME)</sub>

Såfremt tekstenhederne ikke kan separeres skal de annoteres samlet:
- <u>10. og 12. marts</u><sub>(DATETIME)</sub>
#### QUANTITY
Meningsfulde mængder, såsom valutaer. Enheden (eks. valutaen) skal inkluderes i ordstrækningen.
- <u>37.5 millioner kroner</u><sub>(QUANTITY)</sub>
- <u>375 euro</u><sub>(QUANTITY)</sub>
- <u>4267 SEK</u><sub>(QUANTITY)</sub>
- <u>1000 kilo</u><sub>(QUANTITY)</sub>
#### MISC
Andre tekstenheder der kan anses som enheder, såsom varemærker, produkter, kunstværker, begivenheder osv. Alle kunstigt fremstillede ting betragtes som produkter. Dette kan også inkludere mere abstrakte tekstenheder såsom taler, radioprogrammer, programmeringssprog, kontrakter, lovgivning og teorier (såfremt de er navngivet).

Brands er produkter (or derfor MISC), når de henviser til et produkt eller en produktlinje, men organisationer (ORG), når de henviser til handlende eller producerende instanser.
- <u>Lego</u><sub>(MISC)</sub> er et af verdens mest populære legetøjsmærker
- Har du set <u>Mona Lisa</u><sub>(MISC)</sub>?
- Jeg glæder mig til <u>Roskilde Festival</u><sub>(MISC)</sub>!
### Relationer
Hvis nogle tekstenheder refererer til den samme underliggende tekstenhed gennem forskellige formuleringer, skal du oprette en relation mellem tekstenhederne:
1. Klik på den seneste omtale (f.eks. "John Smith"), og klik herefter på _Create relation between regions_-knappen, som er vist med et link-ikon i højre side af interfacet.
2. Find forekomsten af den anden omtale (f.eks. "Smith, John") og klik på den. Du burde nu se en relation mellem de to tekstenheder, markeret med en pil.

__OBS:__ Du skal kun oprette en relation, såfremt tekstenhederne indeholder <u>samme mængde information</u>. Eksempelvis er "Jørgen Hansen" og "Jørgen" ikke lige informative, og afhængigt af konteksten kan førstnævnte ses som en direkte identifikator og sidstnævnte en indirekte identifikator.

## Trin 2: Maskering
I trin 2 skal du for hver tekstenhed markeret i trin 1 angive, om denne skal __maskeres__, såfremt den kan bruges til __direkte- eller indirekte at identificere personer__, som indgår i teksten.

Mere præcist skal du for hver tekstenhed annoteret i trin 1 angive __identifikator-type__, som består af følgende typer:

<table border="1">
  <tr>
    <th bgcolor="#dddddd">Identifikator-type</th>
    <th bgcolor="#dddddd">Beskrivelse</th>
  </tr>
    <td>DIREKTE</td>
    <td>Såfremt tekstenheden udgør direkte og utvetydigt identificerende information, skal du sætte kryds ved DIREKTE.</td>
  <tr>
    <td>KVASI</td>
    <td>Såfremt tekstenheden udgør indirekte identificerende information, der i kombination med anden viden kan identificere en person, skal du sætte kryds ved KVASI. Indirekte identfikatorer kaldes også for kvasi-identifikatorer</td>
  </tr>
</table>

Når du er gået til trin 2 vil du kunne se to nye kategorier i højre side: __DIREKTE__ og __KVASI__. Såfremt tekstenheden er en direkte- eller kvasi identifikator, skal du ændre kategorien ved først at klikke på tekstenheden, så den lyser op, og derefter vælge den nye kategori. Tekstenheden vil nu blive maskeret.

Eksempel:

<p>Dette er en <span style="background-color: red;"><u>direkte identifikator</u><sub>(PERSON)</sub></span>, og dette er en <span style="background-color: green;"><u>kvasi identifikator</u><sub>(MISC)</sub></span>.</p>

Efter at have valgt identifikator-type:

<p>Dette er en <span style="background-color: #000000; color: #000000;">direkte identifikator</span>, og dette er en <span style="background-color: #8b8b8b; color: #8b8b8b;">kvasi identifikator</span>.</p>


__OBS:__ Hvis tekstenheden ikke er en direkte- eller kvasi identifikator, skal du ikke ændre på tekstenheden!

__Hvad er en person?__

En person er her defineret som en naturlig person, hvilket omfatter alle __levende, menneskelige individer__. Du skal være opmærksom på, at dokumentet først er anonymiseret idet identiteten af <u>alle</u> personer, som optræder i teksten, er beskyttet.

Udover identifikator-type skal du angive, hvorvidt informationen i tekstenheden tilhører en __fortrolig kategori__. Du kan angive hvorvidt informationen falder i en af de fortrolige kategorier ved af sætte kryds ud fra kategorien til venstre i interfacet. Informationen er fortrolig såfremt hvis den falder under en af følgende kategorier:

<table border="1">
  <tr>
    <th bgcolor="#dddddd">Fortrolig kategori</th>
    <th bgcolor="#dddddd">Beskrivelse</th>
  </tr>
    <td>BELIEF</td>
    <td>Religiøse eller filosofiske overbevisninger</td>
  <tr>
    <td>POLITICS</td>
    <td>Politiske holdninger eller tilhørsforhold</td>
  </tr>
  <tr>
    <td>SEX</td>
    <td>Sexuel orientering elle andre oplysninger om sexliv</td>
  </tr>
  <tr> 
    <td>ETHNIC</td>
    <td>Etnisk oprindelse</td>
  </tr>
  <tr>
    <td>HEALTH</td>
    <td>Sundheds-, genetisk- og biometrisk data, herunder sensitive sundhedsmæssige oplysninger som misbrug o. lign.</td>
  </tr>
</table>

Hvis informationen ikke er fortrolig, skal du ikke sætte et kryds.
### Eksempler på identifikatorer
#### Direkte identifikatorer
Information, som direkte og utvetydigt kan identificere en person. Typiske eksempler er personnavne (herunder kælenavne/aliasser og brugernavne), CPR-numre, telefonnumre, e-mailadresser, adresser, kontooplysninger mm.

Et personnavn kan enten være en direkte identifikator (hvis det er det fulde navn på en person) eller en indirekte identifikator, afhængigt af konteksten. Eksempelvis er nogle navne så udbredte, at de ikke leder til direkte identifikation af en person. Ligeledes kan diverse koder være enten direkte- eller indirekte identifikatorer, alt efter om de entydigt refererer til den person, der skal beskyttes, eller ej.
#### Indirekte (kvasi) identifikatorer
Information, der i sig selv ikke kan bruges til at identificerer en person, men som i kombination med andre indirekte identifikatorer og/eller baggrundsviden kan lede til identificering af en person. Indirekte identifikatorer kan eksempelvis være demografiske oplysninger (“en 72-årig mand”) eller angivelser af tid og/eller sted (“den 6. februar i Sevilla”). En kombination af fødselsdato, køn og erhverv vil typisk gøre det muligt at finde frem til en persons identitet.

For at indirekte identifikatorer kan lede til identificering af en person, skal disse ses i sammenhæng med "offentligt tilgængelig viden"; oplysninger, som man med rimelighed kan forvente, at en ekstern person enten allerede ved om individet, eller vil kunne finde ud af. Med andre ord bør du stille dig selv følgende spørgsmål: Hvis jeg ville finde ud af, hvem personen i dokumentet er, ville jeg så være i stand til at kombinere disse informationer med andre kilder (såsom nyhedsartikler, sociale medier, databaser, osv.)? Og er disse oplysninger da nok til at re-identificere individet med ingen eller lav tvetydighed? _I langt de fleste tilfælde behøver du ikke at undersøge disse kilder — din umiddelbare intuition er nok._ Hvis du vurderer, at kombinationen af indirekte identifikatorer samt offentligt tilgængelig viden ikke er tilstrækkelig for at re-identificere individet med ingen eller lav usikkerhed, kan tekstenhederne da ikke anses for at være indirekte identifikatorer.

Hvad er offentligt tilgængelig viden? I praksis omfatter det alt, der kan findes ved at søge på internettet. Dog er en del af de dokumenter, du skal annotere, hentet fra internettet, og i disse tilfælde er dokumentet selv ikke omfattet i definition af offentligt tilgængelig viden; her skal man forestille sig, at dokumentet <u>ikke er tilgængeligt på internettet</u>.

Som tommelfingerregel bør uforanderlige personlige attributter (såsom fødselsdato), som kan kendes eller forefindes ved eksterne kilder/databaser, betragtes som indirekte identifikatorer. Andre personlige attributter kan betragtes som indirekte identifikatorer afhængigt af sandsynligheden for, at disse oplysninger kan kendes eller forefindes ved eksterne kilder/databaser. Eksempelvis kan nuværende bopæl eller dato for en hospitalsindlæggelse indirekte identificere en person, men antallet af gange man har handlet i supermarkedet på en uge vil sjældent kunne anses som en indirekte identifikator. Kun meget generelle attributter, som kendetegner et stort antal personer (f.eks. fødeland), ignoreres, da de ofte ikke markant øger muligheden for re-identifikation af en person. Dog er konteksten yderst vigtig, herunder hvilke andre indirekte identifikatorer der findes i dokumentet. Jo flere, der findes, og jo mere konkrete oplysninger de giver, desto større er sandsynligheden for, at de tilsammen kan muliggøre re-identifikation af en person.

## Trin 3: Sidste gennemgang
Når du er færdig med trin 1 og 2 skal du til sidst gennemgå dokumentet for at sikre, at du ikke har overset nogle tekstenheder. Forestil dig, at alle direkte- og indirekte identifikatorer, som du har annoteret, er maskeret (streget over). Læs dokumentet igennem igen - ville du nu være i stand til at re-identificere nogle personer? Hvis ja, skal du gennemgå trin 1 og 2 igen, indtil dette ikke længere er tilfældet.

Når du er færdig skal du gemme dokumentet ved at klikke på Submit-knappen. Du kan nu fortsætte til næste dokument.

# Eksporter annoteringer
For at eksportere dine annoteringer skal du klikke på Export-knappen i højre hjørne når projektet er åbent i Label Studio. Gem som JSON-format.