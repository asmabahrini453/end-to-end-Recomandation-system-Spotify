��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
؇
ConstConst*
_output_shapes	
:�*
dtype0	*��
value��B��	�"��                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      
��
Const_1Const*
_output_shapes	
:�*
dtype0*��
value��B���BLorde|||RoyalsBM83|||Midnight CityBImagine Dragons|||RadioactiveBThe Lumineers|||Ho HeyB"Of Monsters and Men|||Little TalksB"Daft Punk|||Get Lucky - Radio EditBAvicii|||Wake Me UpB:Macklemore & Ryan Lewis|||Can't Hold Us - feat. Ray DaltonBBastille|||PompeiiBRobin Thicke|||Blurred LinesB@Fun.|||We Are Young (feat. Janelle Monáe) - feat. Janelle MonaeB#Foster The People|||Pumped up KicksB!Nirvana|||Smells Like Teen SpiritBMGMT|||KidsBRadiohead|||CreepB#The Temper Trap|||Sweet DispositionBPassenger|||Let Her GoBDaft Punk|||Get LuckyBOneRepublic|||Counting StarsBImagine Dragons|||It's TimeB2Macklemore & Ryan Lewis|||Thrift Shop - feat. WanzBAWOLNATION|||SailB,Clean Bandit|||Rather Be (feat. Jess Glynne)B$Gotye|||Somebody That I Used To KnowBJAY Z|||Ni**as In ParisBBlur|||Song 2BThe Killers|||Mr. BrightsideB Carly Rae Jepsen|||Call Me MaybeBImagine Dragons|||DemonsBSia|||ChandelierBOasis|||WonderwallB!Drake|||Hold On, We're Going HomeB!Arctic Monkeys|||Do I Wanna Know?B;Icona Pop|||I Love It (feat. Charli XCX) - Original VersionBMumford & Sons|||I Will WaitBRihanna|||We Found LoveBPassion Pit|||Take a WalkBJourney|||Don't Stop Believin'BMark Ronson|||Uptown FunkBCalvin Harris|||SummerB$Two Door Cinema Club|||What You KnowBBon Iver|||Skinny LoveBDaft Punk|||Instant CrushB%The White Stripes|||Seven Nation ArmyBVance Joy|||RiptideB"Daft Punk|||Lose Yourself to DanceB*Florence + The Machine|||Dog Days Are OverBNirvana|||Come As You AreBColdplay|||Viva La VidaBPitbull|||TimberBKaty Perry|||RoarBThe Cure|||Friday I'm In LoveBJustin Timberlake|||MirrorsB#Guns N' Roses|||Sweet Child O' MineB!Bruno Mars|||Locked Out Of HeavenBGorillaz|||Feel Good IncBThe xx|||CrystalisedBPhoenix|||LisztomaniaBDisclosure|||LatchBPhoenix|||1901BSurvivor|||Eye of the TigerBFun.|||Some NightsBJMaroon 5|||Moves Like Jagger - Studio Recording From The Voice PerformanceBPharrell Williams|||HappyBLorde|||TeamBMuse|||UprisingBFranz Ferdinand|||Take Me OutBJohnny Cash|||HurtBR.E.M.|||Losing My ReligionBEllie Goulding|||BurnBJason Mraz|||I'm YoursBAvicii|||Hey BrotherBalt-J|||BreezeblocksB%Florence + The Machine|||Shake It OutB+Mr. Probz|||Waves - Robin Schulz Radio EditBMilky Chance|||Stolen DanceBCHVRCHES|||The Mother We ShareBColdplay|||YellowBThe xx|||IntroBRadiohead|||Karma PoliceBThe Strokes|||Last NiteBLSwedish House Mafia|||Don't You Worry Child (Radio Edit) [feat. John Martin]B%Imagine Dragons|||On Top Of The WorldBMGMT|||Electric FeelBKaty Perry|||Dark HorseB Mumford & Sons|||Little Lion ManB"The Naked And Famous|||Young BloodBCalvin Harris|||Sweet NothingBThe Cranberries|||ZombieBM.I.A.|||Paper PlanesB(OutKast|||Hey Ya! - Radio Mix / Club MixB#The Neighbourhood|||Sweater WeatherBCounting Crows|||Mr. JonesBRihanna|||DiamondsBColdplay|||Fix YouBKanye West|||Black SkinheadB)Edward Sharpe & The Magnetic Zeros|||HomeBEd Sheeran|||The A TeamBEminem|||The MonsterBKanye West|||All Of The LightsBKavinsky|||NightcallBMGMT|||Time to PretendBKaty Perry|||FireworkB"Skrillex|||Bangarang (feat. Sirah)BNirvana|||LithiumBThe Smashing Pumpkins|||1979BToto|||AfricaBLana Del Rey|||Video GamesBSam Smith|||Stay With MeBThe Black Keys|||Lonely BoyBFoals|||My NumberB.Michael Jackson|||Billie Jean - Single VersionBKanye West|||StrongerBLana Del Rey|||Born To DieBFoo Fighters|||EverlongBJohn Legend|||All of MeBMumford & Sons|||The CaveBBon Iver|||HoloceneBThe Killers|||Somebody Told MeBPassion Pit|||SleepyheadBNicki Minaj|||StarshipsB Calvin Harris|||I Need Your LoveB.Two Door Cinema Club|||Something Good Can WorkB(Red Hot Chili Peppers|||Under The BridgeBColdplay|||The ScientistBLorde|||Tennis CourtBBeck|||LoserBEminem|||Love The Way You LieBThe Strokes|||ReptiliaBBon Jovi|||Livin' On A PrayerBGreen Day|||Basket CaseBEd Sheeran|||Thinking Out LoudBModest Mouse|||Float OnBJosé González|||HeartbeatsB%The Black Eyed Peas|||I Gotta FeelingBCapital Cities|||Safe And SoundBKings Of Leon|||Sex on FireBThe xx|||IslandsBAvicii|||Levels - Radio EditB$Meghan Trainor|||All About That BassBKings Of Leon|||Use SomebodyBThe xx|||AngelsB"Lana Del Rey|||Young And BeautifulBColdplay|||A Sky Full of StarsB2Pharrell Williams|||Happy - From Despicable Me 2""BMagic!|||RudeBJames Blake|||RetrogradeBAdele|||Someone Like YouBFoo Fighters|||The PretenderB"The Rolling Stones|||Gimme ShelterBPixies|||Where Is My Mind?BFoo Fighters|||Best Of YouB%Guns N' Roses|||Welcome To The JungleBJohn Newman|||Love Me AgainBDaft Punk|||One More TimeB'Red Hot Chili Peppers|||CalifornicationBWeezer|||Island In The SunBThe Goo Goo Dolls|||IrisBLMFAO|||Party Rock AnthemBwill.i.am|||Scream & ShoutB+Eminem|||Lose Yourself - Soundtrack VersionB Vampire Weekend|||A-Punk (Album)BColdplay|||ParadiseBMassive Attack|||TeardropBPhoenix|||If I ever feel betterBP!nk|||Just Give Me a ReasonBNico & Vinz|||Am I WrongBMaroon 5|||PayphoneBNo Doubt|||Don't SpeakBU2|||Beautiful DayBGrouplove|||Tongue TiedBIggy Azalea|||FancyBThe Cure|||Just Like HeavenBJack Johnson|||Better TogetherBThe Script|||Hall of FameBColdplay|||ClocksBFleetwood Mac|||Go Your Own WayBAlex Clare|||Too CloseBFrank Ocean|||Thinkin Bout YouB Of Monsters and Men|||Dirty PawsB"Rihanna|||Only Girl (In The World)BColdplay|||MagicBRadiohead|||No SurprisesBMuse|||MadnessBFoster The People|||HoudiniB+The Rolling Stones|||Sympathy For The DevilBArctic Monkeys|||R U Mine?BSantigold|||Disparate YouthBFlo Rida|||Good FeelingBMaroon 5|||One More NightBThe Lumineers|||Stubborn LoveB!Lana Del Rey|||Summertime SadnessBSoundgarden|||Black Hole SunB8Macklemore & Ryan Lewis|||Same Love - feat. Mary LambertB&Empire Of The Sun|||Walking On A DreamBBoston|||More Than a FeelingBSnow Patrol|||Chasing CarsB*Calvin Harris|||Feel so Close - Radio EditB2JAY Z|||Empire State Of Mind [Jay-Z + Alicia Keys]BWeezer|||Buddy HollyBKanye West|||POWERBEagle-Eye Cherry|||Save TonightB#David Guetta|||Titanium (feat. Sia)BBruno Mars|||TreasureBVan Morrison|||Brown Eyed GirlBEarth, Wind & Fire|||SeptemberB,Kendrick Lamar|||Bitch, Don’t Kill My VibeBAriana Grande|||ProblemBZLana Del Rey|||Summertime Sadness [Lana Del Rey vs. Cedric Gervais] - Cedric Gervais RemixBMuse|||StarlightBThe Black Keys|||Tighten UpB"Peter Bjorn And John|||Young FolksB6Arctic Monkeys|||I Bet You Look Good On The DancefloorBRadiohead|||Paranoid AndroidBJAY Z|||Holy GrailBJet|||Are You Gonna Be My GirlBTame Impala|||ElephantB blink-182|||All The Small ThingsB2Snoop Dogg|||Young, Wild & Free (feat. Bruno Mars)BAdele|||Rolling In The DeepBKanye West|||Gold DiggerBSam Smith|||Money On My MindB!Simon & Garfunkel|||Mrs. RobinsonBHaim|||The WireBMiley Cyrus|||Wrecking BallBHozier|||Take Me To ChurchB!A Great Big World|||Say SomethingB9Arctic Monkeys|||Why'd You Only Call Me When You're High?BGrimes|||GenesisB#Lynyrd Skynyrd|||Sweet Home AlabamaBJAY Z|||No Church In The WildBDaft Punk|||Giorgio by MoroderBPearl Jam|||Even FlowB.Rudimental|||Feel The Love - feat. John NewmanBDaft Punk|||Doin' it RightBU2|||With Or Without YouB*Jason Derulo|||Talk Dirty (feat. 2 Chainz)BAlabama Shakes|||Hold OnBGorillaz|||Clint EastwoodB@Lilly Wood and The Prick|||Prayer in C - Robin Schulz Radio EditBJustin Timberlake|||SexyBackB!The Killers|||When You Were YoungB'The Postal Service|||Such Great HeightsBCalvin Harris|||BlameB9The Verve|||Bitter Sweet Symphony - 2004 Digital RemasterBLMFAO|||Sexy And I Know ItBBand of Horses|||The FuneralBGuns N' Roses|||Paradise CityBKe$ha|||Die YoungBHouse Of Pain|||Jump AroundBMiley Cyrus|||We Can't StopBVampire Weekend|||Diane YoungBGrimes|||OblivionBFoster The People|||Helena BeatBJessie J|||Bang BangBThe Cure|||Boys Don't CryBArcade Fire|||ReflektorB5Death Cab for Cutie|||I Will Follow You Into The DarkBSmash Mouth|||All StarBZedd|||ClarityBThe Strokes|||SomedayBQueen|||Don't Stop Me NowB,Florence + The Machine|||You've Got The LoveBLinkin Park|||In The EndBSpin Doctors|||Two PrincesB*Michael Jackson|||Beat It - Single VersionBTrain|||Drive ByBRihanna|||S&MBIggy Pop|||The PassengerBA$AP Rocky|||F**kin' ProblemsBPearl Jam|||AliveBBlackstreet|||No DiggityB%PSY|||Gangnam Style (강남스타일)B&Faul & Wad Ad|||Changes - Original MixBLady Gaga|||Born This WayB.Justin Timberlake|||Suit & Tie featuring JAY ZBKe$ha|||TiK ToKBLa Roux|||BulletproofB&Joy Division|||Love Will Tear Us ApartB"Bright Eyes|||First Day Of My LifeBEd Sheeran|||I See FireB(David Guetta|||Memories (feat. Kid Cudi)BDaft Punk|||Around The WorldB!Phil Collins|||In The Air TonightBRadiohead|||High And DryBSia|||Breathe MeB!The Cardigans|||My Favourite GameBFlo Rida|||WhistleB Flo Rida|||Wild Ones (feat. Sia)B3Tears For Fears|||Everybody Wants To Rule The WorldBEurope|||The Final CountdownBFleetwood Mac|||DreamsB#Daft Punk|||Give Life Back to MusicBYeah Yeah Yeahs|||MapsB!Bruno Mars|||Just The Way You AreBJennifer Lopez|||On The FloorBHaim|||ForeverBKiesza|||HideawayB)Daft Punk|||Harder Better Faster StrongerBTLC|||No ScrubsBOutKast|||Ms. JacksonBThe xx|||VCRB(Two Door Cinema Club|||Undercover MartynBNaughty Boy|||La La LaBPitbull|||Give Me EverythingBSupergrass|||AlrightB&American Authors|||Best Day Of My LifeBDire Straits|||Sultans Of SwingBMetallica|||Enter SandmanBa-ha|||Take On MeBLady Gaga|||ApplauseB(The Smashing Pumpkins|||Tonight, TonightBNatalie Imbruglia|||TornBCalvin Harris|||Under ControlBThe 1975|||ChocolateBAvicii|||You Make MeBTrain|||Hey, Soul SisterBFoo Fighters|||My HeroB*The Proclaimers|||I'm Gonna Be (500 Miles)BGorillaz|||On Melancholy HillBRobyn|||Dancing On My OwnBMaroon 5|||SugarBWalk the Moon|||Anna SunBMaroon 5|||MapsBKeane|||Somewhere Only We KnowBDJ Snake|||Turn Down for WhatB'Arctic Monkeys|||When The Sun Goes DownB#Disclosure|||You & Me - Flume RemixBCrystal Castles|||Not In LoveBNeon Indian|||Polish GirlBAlanis Morissette|||IronicB Sam Smith|||I'm Not The Only OneB$Of Monsters and Men|||Mountain SoundB50 Cent|||In Da ClubBBon Iver|||PerthBAmy Winehouse|||Back To BlackBJackson 5|||I Want You BackBBryan Adams|||Summer Of '69B0Katy Perry|||California Gurls - feat. Snoop DoggBGuns N' Roses|||November RainB.Rage Against The Machine|||Killing In The NameBEchosmith|||Cool KidsBWashed Out|||Feel It All AroundB'Arctic Monkeys|||Fluorescent AdolescentBMetronomy|||The LookBKAvicii|||I Could Be The One [Avicii vs Nicky Romero] - Nicktim - Radio EditB#David Guetta|||Titanium - feat. SiaBThe Kooks|||NaiveBHaim|||FallingBMC Hammer|||U Can't Touch ThisB&Ellie Goulding|||Anything Could HappenB$New Radicals|||You Get What You GiveB!Electric Guest|||This Head I HoldBTaio Cruz|||DynamiteBBruno Mars|||The Lazy SongBHaim|||Don't Save MeBBEllie Goulding|||Love Me Like You Do - From Fifty Shades Of Grey""BThe Killers|||HumanBIncubus|||DriveBBeach House|||MythB-The Killers|||All These Things That I've DoneBLana Del Rey|||Blue JeansBJessie J|||Price TagB-Tame Impala|||Feels Like We Only Go BackwardsBPretty Lights|||Finally MovingB)Bonnie Tyler|||Total Eclipse of the HeartBJAY Z|||99 ProblemsB3The Smashing Pumpkins|||Bullet With Butterfly WingsBDrake|||Take CareBCharli XCX|||Boom ClapBCalvin Harris|||OutsideBMetric|||Help I'm AliveBMaroon 5|||She Will Be LovedBFar East Movement|||Like A G6BEd Sheeran|||SingB-Armin van Buuren|||This Is What It Feels LikeBParamore|||Misery BusinessBEminem|||Not AfraidBRadiohead|||Fake Plastic TreesB Bill Withers|||Ain't No SunshineBBlur|||Girls And BoysB#Bon Jovi|||You Give Love A Bad NameBPassion Pit|||Little SecretsBVampire Weekend|||UnbelieversBAloe Blacc|||The ManBThe Cranberries|||LingerBNeil Young|||Heart Of GoldBMaroon 5|||This LoveBAmy Winehouse|||RehabBKanye West|||CliqueBMiike Snow|||AnimalB Bruno Mars|||When I Was Your ManB"Justin Timberlake|||Rock Your BodyBSoft Cell|||Tainted LoveBDr. Dre|||The Next EpisodeBThe Shins|||Simple SongBBeyoncé|||Crazy in LoveB'Dexys Midnight Runners|||Come On EileenBEd Sheeran|||Give Me LoveBArctic Monkeys|||Mardy BumB,Creedence Clearwater Revival|||Fortunate SonBBirdy|||Skinny LoveB)Radiohead|||Everything In Its Right PlaceB6Taylor Swift|||We Are Never Ever Getting Back TogetherBMichael Jackson|||ThrillerBNine Inch Nails|||CloserBDaft Punk|||ContactBFoo Fighters|||Learn to FlyB%Empire Of The Sun|||We Are the PeopleBPixies|||Here Comes Your ManBJustice|||D.A.N.C.E.BLykke Li|||I Follow RiversB5John Mayer|||Free Fallin' - Live at the Nokia TheatreBMen At Work|||Down UnderBSystem Of A Down|||Chop Suey!B,The White Stripes|||Fell In Love With A GirlBDaughter|||YouthB-Green Day|||Good Riddance [Time Of Your Life]BSemisonic|||Closing TimeBNirvana|||In BloomBalt-J|||TessellateB)Buffalo Springfield|||For What It's WorthB The Notorious B.I.G.|||HypnotizeBDjango Django|||DefaultBSublime|||SanteriaBBeyoncé|||Drunk in LoveBKanye West|||Bound 2B
P!nk|||TryBBombay Bicycle Club|||ShuffleBMadonna|||Like A PrayerBThe Temptations|||My GirlB/Otis Redding|||(Sittin' On) The Dock Of The BayBDrake|||Started From The BottomBVampire Weekend|||StepBBloc Party|||BanquetB#Damien Rice|||The Blower's DaughterB'SBTRKT|||Wildfire (feat. Little Dragon)BFKid Cudi|||Pursuit Of Happiness - Extended Steve Aoki Remix (Explicit)BMartin Garrix|||AnimalsB1Kelly Clarkson|||Stronger (What Doesn't Kill You)BNirvana|||Heart-Shaped BoxBDisclosure|||White NoiseBKaty Perry|||Teenage DreamB<Whitney Houston|||I Wanna Dance with Somebody (Who Loves Me)BTracy Chapman|||Fast CarB blink-182|||What's My Age Again?BABBA|||Dancing QueenBJimmy Eat World|||The MiddleB&Sinead O'Connor|||Nothing Compares 2 UBDr. Dre|||Still D.R.E.B9Pulp|||Common People - Full Length Version; Album VersionBR.E.M.|||Everybody HurtsB*Blue Oyster Cult|||(Don't Fear) The ReaperB"The Police|||Every Breath You TakeBEric Clapton|||Tears In HeavenBMuse|||Undisclosed DesiresBLana Del Rey|||National AnthemB&Queens Of The Stone Age|||No One KnowsBJackson 5|||ABCBAzealia Banks|||212BAvicii|||Addicted To YouBLady Gaga|||Poker FaceBTrain|||Drops of JupiterB)Foster The People|||Call It What You WantB(The Chemical Brothers|||Hey Boy Hey GirlBBruno Mars|||GrenadeBMumford & Sons|||Winter WindsBFoo Fighters|||Monkey WrenchBCHVRCHES|||RecoverBBR.E.M.|||It's the End of the World As We Know It (And I Feel Fine)B"Spice Girls|||Wannabe - Radio EditB)The Cinematic Orchestra|||To Build A HomeBLady Gaga|||Bad RomanceBDisclosure|||F For YouB!Yeah Yeah Yeahs|||Heads Will RollBGnarls Barkley|||CrazyBGeorge Ezra|||BudapestBMazzy Star|||Fade Into YouBFoo Fighters|||Times Like TheseBSheppard|||GeronimoBPearl Jam|||JeremyB$LCD Soundsystem|||Dance Yrself CleanB'Lenny Kravitz|||Are You Gonna Go My WayBColdplay|||TroubleBColdplay|||Don't PanicBEd Sheeran|||Don'tBBlack Sabbath|||ParanoidB3David Guetta|||Lovers on the Sun (feat. Sam Martin)BLinkin Park|||NumbB&College & Electric Youth|||A Real HeroBCapital Cities|||Safe and SoundBAriana Grande|||Break FreeB"Snoop Dogg|||Drop It Like It's HotBPitbull|||Feel This MomentBDaft Punk|||Fragments of TimeBEminem|||The Real Slim ShadyBDaft Punk|||WithinBRhye|||OpenB4 Non Blondes|||What's Up?BOasis|||Champagne SupernovaB$Amy Winehouse|||You Know I'm No GoodBThe Weeknd|||Wicked GamesB)The Animals|||The House Of The Rising SunBThe Smashing Pumpkins|||TodayBNONONO|||Pumpin BloodBAretha Franklin|||RespectB&Bruce Springsteen|||Born in the U.S.A.BThe Outfield|||Your LoveBCFall Out Boy|||My Songs Know What You Did In The Dark (Light Em Up)BPhosphorescent|||Song For ZulaBFoo Fighters|||WalkBThe Shins|||New SlangBJason Mraz|||I Won't Give UpBMS MR|||HurricaneBThe Notorious B.I.G.|||JuicyBExtreme|||More Than WordsB'Bruce Springsteen|||Dancing in the DarkB"Sixpence None The Richer|||Kiss MeBMuse|||Supermassive Black HoleB+David Guetta|||Dangerous (feat. Sam Martin)BBon Jovi|||It's My LifeBGrizzly Bear|||Two WeeksB!Usher|||DJ Got Us Fallin' in LoveBJAY Z|||OtisBLady Gaga|||TelephoneB*The Naked And Famous|||Punching In A DreamBRihanna|||Where Have You BeenB!Mumford & Sons|||White Blank PageBBon Iver|||TowersBDaft Punk|||BeyondBU2|||OneB One Direction|||Story of My LifeB%A Tribe Called Quest|||Can I Kick It?BThe Knife|||HeartbeatsB.Creedence Clearwater Revival|||Bad Moon RisingBElton John|||Your SongBFeist|||1234BKendrick Lamar|||m.A.A.d cityBBruno Mars|||Marry YouB-Bob Marley & The Wailers|||Could You Be LovedBLana Del Rey|||RadioB*The Black Eyed Peas|||The Time (Dirty Bit)BKaiser Chiefs|||RubyBThe Vaccines|||If You WannaBBroken Bells|||The High RoadBFleetwood Mac|||The ChainB%The Dandy Warhols|||Bohemian Like YouBOwl City|||FirefliesBDuke Dumont|||I Got UBBon Iver|||CalgaryB*Skrillex|||Scary Monsters And Nice SpritesB James Blake|||Limit To Your LoveB"Tove Lo|||Stay High - Habits RemixBGorillaz|||DAREBTiësto|||WastedB#Third Eye Blind|||Semi-Charmed LifeBMumford & Sons|||BabelBParamore|||Still Into YouBBritney Spears|||Work B**chBGreen Day|||When I Come AroundB#Red Hot Chili Peppers|||Scar TissueB*Simple Minds|||Don't You (Forget About Me)BKygo|||FirestoneBLou Reed|||Perfect DayBThe xx|||ChainedB)Katy Perry|||Last Friday Night (T.G.I.F.)B9David Guetta|||When Love Takes Over (feat. Kelly Rowland)BNeon Trees|||Everybody TalksBThe xx|||Heart Skipped A BeatB$The Kooks|||She Moves In Her Own WayB"Calvin Harris|||Thinking About YouBEd Sheeran|||Lego HouseB'The Offspring|||The Kids Aren't AlrightB Metallica|||Nothing Else MattersBSwedish House Mafia|||GreyhoundB<David Guetta|||Shot me Down (feat. Skylar Grey) - Radio EditBThe National|||I Need My GirlBSystem Of A Down|||ToxicityB(Ellie Goulding|||Lights - Single VersionB"The Black Keys|||Everlasting LightBQueen|||Bohemian RhapsodyB8Snoop Dogg|||Sweat (Snoop Dogg vs. David Guetta) [Remix]BBlondie|||Heart Of GlassBDaft Punk|||The Game of LoveB!Fleet Foxes|||White Winter HymnalBAl Green|||Let's Stay TogetherBDaft Punk|||TouchBBen Howard|||Keep Your Head UpBThe Fray|||How to Save a LifeBJessie J|||DominoB/Lykke Li|||I Follow Rivers - The Magician RemixBJason Derulo|||TrumpetsBMatt and Kim|||DaylightB!The Verve|||Bitter Sweet SymphonyB$Florence + The Machine|||Cosmic LoveBWiz Khalifa|||Black And YellowBChumbawamba|||TubthumpingB,Daryl Hall & John Oates|||You Make My DreamsB&Shakira|||Can't Remember to Forget YouBColdplay|||Princess of ChinaB4Gym Class Heroes|||Stereo Hearts - feat. Adam LevineBNo Doubt|||Just A GirlB,Major Lazer|||Lean On (feat. MØ & DJ Snake)BThe xx|||SunsetBPhantogram|||Don't MoveB5Stevie Wonder|||Signed, Sealed, Delivered (I'm Yours)B"Lauryn Hill|||Doo Wop (That Thing)BScorpions|||Wind Of ChangeB&Bombay Bicycle Club|||Always Like ThisBBloc Party|||HelicopterBThe Fratellis|||Chelsea DaggerBBon Iver|||FlumeBThe Script|||BreakevenBCaribou|||OdessaBTegan And Sara|||CloserBNeon Trees|||AnimalB"Red Hot Chili Peppers|||Can't StopBDire Straits|||Walk Of LifeBDaft Punk|||MotherboardBAlesso|||Heroes (we could be)Balt-J|||FitzpleasureBalt-J|||Something GoodB$Britney Spears|||Till the World EndsBThe Whitest Boy Alive|||BurningBLittle Dragon|||Ritual UnionBBloodhound Gang|||The Bad TouchB!Led Zeppelin|||Stairway To HeavenBFrank Ocean|||LostBalt-J|||Hunger Of The PineBRihanna|||What's My Name?B8Fergie|||A Little Party Never Killed Nobody (All We Got)BSelena Gomez|||Come & Get ItB Elvis Presley|||Suspicious MindsBblink-182|||I Miss YouBBeirut|||Santa FeBM.I.A.|||Bad GirlsBElton John|||Tiny DancerBJake Bugg|||Lightning BoltBSublime|||What I GotBParamore|||Ain't It FunB&Skrillex|||First Of The Year (Equinox)B@David Guetta|||Sexy Bitch (feat. Akon) - Featuring Akon;explicitB+Bruce Springsteen|||Streets of PhiladelphiaBThe Cure|||Close To MeBUsher|||Yeah!B
M83|||WaitB@C & C Music Factory|||Gonna Make You Sweat (Everybody Dance Now)BBeyoncé|||HaloB(Of Monsters and Men|||King And LionheartBMumford & Sons|||Sigh No MoreBEagles|||Hotel CaliforniaBTwo Door Cinema Club|||SunB'Wham!|||Last Christmas - Single VersionBKanye West|||HomecomingBBastille|||FlawsBNirvana|||Rape MeBJohn Mayer|||Who SaysBMuse|||Time Is Running OutBKanye West|||MonsterBSigur Rós|||HoppípollaBWarren G|||RegulateBPurity Ring|||FineshrineBDaft Punk|||Digital LoveBPaul Simon|||You Can Call Me AlBJason Mraz|||LuckyBToto|||Hold the LineBTelepopmusik|||BreatheB#Kendrick Lamar|||Backseat FreestyleBMacy Gray|||I TryB1The Smiths|||There Is A Light That Never Goes OutB The Black Keys|||Howlin' For YouB Lou Reed|||Walk on the Wild SideBMuse|||Knights Of CydoniaBDon Omar|||Danza KuduroBKodaline|||All I WantBParamore|||The Only ExceptionB%Crowded House|||Don't Dream It's OverBThe Breeders|||CannonballBEmpire Of The Sun|||AliveBElvis Presley|||Jailhouse RockBBlondie|||Call MeBLana Del Rey|||Off To The RacesBIggy Pop|||Lust For LifeBJustin Timberlake|||SeñoritaBBeyoncé|||Love On TopB+Beyoncé|||Single Ladies (Put a Ring on It)B$Backstreet Boys|||I Want It That WayBMaroon 5|||AnimalsB Beyoncé|||Run the World (Girls)BIce Cube|||It Was A Good DayBEminem|||'Till I CollapseBKe$ha|||We R Who We RBCrazy Town|||ButterflyBMarvin Gaye|||Sexual HealingBFlo Rida|||I CryBLorde|||Glory And GoreBOwl City|||Good TimeBMuse|||HysteriaB)Thirty Seconds To Mars|||Kings and QueensBMetallica|||Master Of PuppetsB%Prince & The Revolution|||Purple RainB)Katrina & The Waves|||Walking On SunshineB(The Offspring|||You're Gonna Go Far, KidBThe xx|||InfinityBBen Howard|||Old PineBA$AP Rocky|||Wild for the NightBBruce Springsteen|||Born to RunB8The Smiths|||This Charming Man (2008 Remastered Version)BThe Smashing Pumpkins|||DisarmB-Stevie Wonder|||Superstition - Single VersionBLana Del Rey|||Dark ParadiseBThe Script|||For the First TimeB!Foster The People|||Coming of AgeB!Walk the Moon|||Shut Up and DanceBKanye West|||HeartlessBArcade Fire|||AfterlifeBMarvin Gaye|||Let's Get It OnB)Dusty Springfield|||Son Of A Preacher ManB%Rick Astley|||Never Gonna Give You UpB Radical Face|||Welcome Home, SonBDamien Rice|||CannonballBTaylor Swift|||Shake It OffB4Flo Rida|||Club Can't Handle Me - feat. David GuettaB*Stone Temple Pilots|||Interstate Love SongBJeff Buckley|||HallelujahBThe Wanted|||Glad You CameBLenny Kravitz|||Fly AwayBNicki Minaj|||Super BassB/Florence + The Machine|||What The Water Gave MeBDon McLean|||American PieB#Plain White T's|||Hey There DelilahBAdele|||Rolling in the DeepBMoby|||PorcelainBNero|||PromisesBalt-J|||Dissolve MeBCThe Mojos|||One Day / Reckoning Song (Wankelmut Remix) - Radio EditBThe Roots|||The Seed (2.0)B6The Human League|||Don't You Want Me - 2002 - RemasterBUsher|||ScreamBRadiohead|||JustBWeezer|||Say It Ain't SoBLa Roux|||In For The KillB$The Killers|||Smile Like You Mean ItBCElton John|||Rocket Man (I Think It's Going To Be A Long Long Time)BBastille|||Bad BloodBEminem|||StanB-David Guetta|||Bad (feat. Vassy) - Radio EditB Ace of Base|||All That She WantsBJAY Z|||Tom FordB#The Black Eyed Peas|||Boom Boom PowBPearl Jam|||Just BreatheBalt-J|||MatildaBKanye West|||MercyBFeist|||MushaboomBArcade Fire|||Rebellion (Lies)BDuffy|||MercyB(Aerosmith|||I Don't Want to Miss a ThingBAerosmith|||Dream OnB Rick Springfield|||Jessie's GirlBKendrick Lamar|||Poetic JusticeBKendrick Lamar|||Money TreesB!Disclosure|||Help Me Lose My MindBAtlas Genius|||TrojansB(Whitney Houston|||I Will Always Love YouB)Noah And The Whale|||L.I.F.E.G.O.E.S.O.N.B3Foster The People|||Don't Stop (Color on the Walls)BHoobastank|||The ReasonBLorde|||Buzzcut SeasonB%Van Halen|||Jump - Remastered VersionB Massive Attack|||Paradise CircusBPortishead|||Glory BoxBNew Order|||Blue MondayB"Loreen|||Euphoria - Single VersionBEminem|||Rap GodBLana Del Rey|||RideBJustice|||CivilizationBR. Kelly|||Ignition (Remix)BSoul Asylum|||Runaway TrainB1Edward Sharpe & The Magnetic Zeros|||40 Day DreamBYoung the Giant|||Cough SyrupBChris Brown|||Don't Wake Me UpBEminem|||Without MeB%Fall Out Boy|||Sugar, We're Goin DownBWu-Tang Clan|||C.R.E.A.M.B4Beastie Boys|||Intergalactic - 2009 Digital RemasterBLady Gaga|||Just DanceBalt-J|||Left Hand FreeBRoute 94|||My LoveBRJD2|||GhostwriterB'Red Hot Chili Peppers|||Dani CaliforniaBIdina Menzel|||Let It GoB1My Chemical Romance|||Welcome To The Black ParadeBDaft Punk|||Something About UsBCrystal Fighters|||At HomeBCyndi Lauper|||Time After TimeBStromae|||PapaoutaiBImagine Dragons|||AmsterdamBRegina Spektor|||SamsonBJungle|||Busy Earnin'BStone Temple Pilots|||PlushB+Kid Cudi|||Pursuit Of Happiness (nightmare)BLorde|||RibsBThe War On Drugs|||Red EyesBKanye West|||Flashing LightsBThe xx|||FictionB'The Script|||The Man Who Can't Be MovedBAvril Lavigne|||ComplicatedBKeane|||Everybody's ChangingBKanye West|||Love LockdownBEminem|||My Name IsBArcade Fire|||The SuburbsBKings Of Leon|||PyroBColdplay|||Speed Of SoundBRUN-DMC|||Walk This WayBFugees|||Ready or NotBBeastie Boys|||SabotageB$Caesars|||Jerk It Out - Original MixB3 Doors Down|||KryptoniteBSimon & Garfunkel|||The BoxerBFrank Ocean|||PyramidsBBilly Joel|||Piano ManBTove Lo|||Habits (Stay High)BFleet Foxes|||MykonosBThe National|||Bloodbuzz OhioBNina Simone|||Feeling GoodB#Alanis Morissette|||You Oughta KnowBCake|||The DistanceBPapa Roach|||Last ResortB$The Black Keys|||Gold On The CeilingBHozier|||Take Me to ChurchB"Justin Timberlake|||Cry Me a RiverBIggy Azalea|||Black WidowB&Britney Spears|||...Baby One More TimeB'Bob Marley & The Wailers|||Is This LoveBNirvana|||All ApologiesB*Cyndi Lauper|||Girls Just Want to Have FunBIcona Pop|||All NightBArctic Monkeys|||ArabellaBSara Bareilles|||BraveB/U2|||I Still Haven't Found What I'm Looking ForB"Christina Perri|||A Thousand YearsB(Band of Horses|||No One's Gonna Love YouB%The Black Eyed Peas|||Meet Me HalfwayBKaty Perry|||Hot N ColdBFoo Fighters|||All My LifeBFaith No More|||EpicB#Calvin Harris|||Bounce - Radio EditBStevie Wonder|||SuperstitionBHot Chip|||Over And OverBFleetwood Mac|||Don't StopB:Creedence Clearwater Revival|||Have You Ever Seen The RainB$Calvin Harris|||We'll Be Coming BackB"Deee-Lite|||Groove Is In The HeartB Dire Straits|||Money For NothingB'Disclosure|||When A Fire Starts To BurnBKanye West|||Touch The SkyBFleetwood Mac|||EverywhereBCulture Club|||Karma ChameleonBRadiohead|||Let DownBThe xx|||ShelterB(One Direction|||What Makes You BeautifulBBat For Lashes|||DanielBJames Blunt|||You're BeautifulBQueen|||We Will Rock YouBColdplay|||In My PlaceBRam Jam|||Black BettyBMumford & Sons|||Awake My SoulBPixies|||HeyBP!nk|||So WhatB KISS|||I Was Made For Lovin' YouB"Vanessa Carlton|||A Thousand MilesB7David Guetta|||She Wolf (Falling to Pieces) [feat. Sia]B(Simon & Garfunkel|||The Sound of SilenceBSia|||Elastic HeartB%The Strokes|||Under Cover of DarknessB(Jason Derulo|||Wiggle (feat. Snoop Dogg)B(Coldplay|||Every Teardrop Is A WaterfallB The Notorious B.I.G.|||Big PoppaBSolange|||Losing YouB!Two Door Cinema Club|||I Can TalkBBritney Spears|||I Wanna GoBArctic Monkeys|||BrianstormB%Mumford & Sons|||Roll Away Your StoneBNeil Young|||Harvest MoonBThe Killers|||RunawaysB-Bob Marley & The Wailers|||Three Little BirdsBRobyn|||Call Your GirlfriendBThe Knack|||My SharonaB"The Black Keys|||Howlin’ For YouBThe Prodigy|||Smack My Bitch UpB/The Who|||Baba O'Riley - Original Album VersionBMadness|||Our HouseBTom Odell|||Another LoveBR.E.M.|||Shiny Happy PeopleBMissy Elliott|||Get Ur Freak OnBPrince & The Revolution|||KissBMetallica|||OneBGrouplove|||ColoursBTaio Cruz|||Break Your HeartBAerosmith|||Cryin'BRhye|||The FallBThe Black Keys|||FeverBBen Howard|||Only LoveB*Rage Against The Machine|||Bulls on ParadeB Franz Ferdinand|||Do You Want ToBImagine Dragons|||Bleeding OutBPassion Pit|||Carried AwayBEvanescence|||Bring Me To LifeBBruno Mars|||It Will RainB,Kylie Minogue|||Can't Get You Out Of My HeadBNorah Jones|||Come Away With MeB$Of Monsters and Men|||Love Love LoveB Bob Dylan|||Like a Rolling StoneBCher|||BelieveBInterpol|||EvilBKasabian|||FireB%Jefferson Airplane|||Somebody to LoveBEmeli Sandé|||Next to MeBJimi Hendrix|||Purple HazeBAnimal Collective|||My GirlsBZThe Weeknd|||Earned It (Fifty Shades Of Grey) - From The Fifty Shades Of Grey" Soundtrack"BAloe Blacc|||I Need A DollarBHaddaway|||What Is LoveB/Tom Petty And The Heartbreakers|||American GirlBRihanna|||FourFiveSecondsBSpandau Ballet|||TrueB4JAY Z|||Run This Town [Jay-Z + Rihanna + Kanye West]BKanye West|||New SlavesBEric Clapton|||LaylaBADavid Guetta|||Where Them Girls At (feat. Nicki Minaj & Flo Rida)BLady Gaga|||JudasB>Violent Femmes|||Blister In The Sun (Remastered Album Version)B)Michael Jackson|||Love Never Felt so GoodBalt-J|||TaroBCults|||Go OutsideBLady Gaga|||AlejandroBM83|||ReunionB&Taylor Swift|||I Knew You Were TroubleBJourney|||Any Way You Want ItBNirvana|||PollyBTom Petty|||Free Fallin'BSt. Lucia|||ElevateBKings Of Leon|||CloserBLynyrd Skynyrd|||Free BirdBEllie Goulding|||LightsB!Noah And The Whale|||5 Years TimeB(Echo And The Bunnymen|||The Killing MoonBCream|||Sunshine Of Your LoveBPhoenix|||EntertainmentBFoster The People|||WasteB!Red Hot Chili Peppers|||OthersideB$Red Hot Chili Peppers|||Give It AwayBThe Wallflowers|||One HeadlightBKanye West|||RunawayBImogen Heap|||Hide And SeekB%The Hives|||Hate To Say I Told You SoBCee Lo Green|||Fuck YouB.David Bowie|||Heroes - 1999 Remastered VersionB Oasis|||Don't Look Back In AngerBRihanna|||Don't Stop The MusicBMuse|||ResistanceBBon Iver|||Blood BankBalt-J|||IntroBNorah Jones|||Don't Know WhyBVampire Weekend|||HolidayBBill Withers|||Lovely DayB!Will Smith|||Gettin' Jiggy Wit ItBLady Gaga|||The Edge Of GloryBThe Cure|||LullabyB*Alexandra Stan|||Mr. Saxobeat - Radio EditBRUN-DMC|||It's TrickyBThe xx|||StarsBStarship|||We Built This CityB"Two Door Cinema Club|||Sleep AloneB"Mumford & Sons|||Hopeless WandererB4Arcade Fire|||Sprawl II (Mountains Beyond Mountains)B)Future Islands|||Seasons (Waiting On You)B The Strokes|||You Only Live OnceBGreen Day|||American IdiotBAlice Cooper|||PoisonBKenny Loggins|||FootlooseBJason Derulo|||The Other SideBCyndi Lauper|||True ColorsBFiona Apple|||CriminalBalt-J|||Every Other FreckleB#Beastie Boys|||Fight For Your RightBArcade Fire|||Wake UpBSufjan Stevens|||ChicagoBRihanna|||Rude BoyB LCD Soundsystem|||All My FriendsBThe Drums|||Let's Go SurfingBThe Clash|||London CallingB%The Kooks|||Junk of the Heart (Happy)BTiësto|||Red LightsBBirdy|||WingsBBon Jovi|||AlwaysB)Guns N' Roses|||Knockin' On Heaven's DoorBColdplay|||SparksB%Death Cab for Cutie|||Soul Meets BodyBChairlift|||BruisesB0Flo Rida|||GDFR (feat. Sage The Gemini & Lookas)BBeck|||Blue MoonB Mumford & Sons|||After The StormBP!nk|||Raise Your GlassBCHVRCHES|||GunBThe Band|||The WeightBThe Prodigy|||BreatheBThe xx|||Night TimeBFoo Fighters|||RopeBGotye|||Eyes Wide OpenBKJohn Legend|||All of Me - (Tiësto's Birthday Treatment Remix) [Radio Edit]BKaty Perry|||Part Of MeBColdplay|||Charlie BrownBThe Cardigans|||LovefoolBThe Smashing Pumpkins|||ZeroBPitbull|||FireballBYoung the Giant|||My BodyBDr. Dre|||Forgot About DreB6OneRepublic|||If I Lose Myself - Alesso vs OneRepublicBFKA twigs|||Two WeeksBMichael Jackson|||BadBThe Goo Goo Dolls|||SlideBThe Police|||RoxanneB(Louis Armstrong|||What A Wonderful WorldB!Diddy - Dirty Money|||Coming HomeB Alien Ant Farm|||Smooth CriminalBKasabian|||Club FootB!Miley Cyrus|||Party In The U.S.A.B Kanye West|||Blood On The LeavesBEminem|||BerzerkB7The Rolling Stones|||You Can't Always Get What You WantBImagine Dragons|||Hear MeB-Michael Jackson|||P.Y.T. (Pretty Young Thing)BRihanna|||UmbrellaBKings Of Leon|||RadioactiveB!The Chemical Brothers|||GalvanizeBKaty Perry|||Wide AwakeBRay LaMontagne|||TroubleBMaroon 5|||Sunday MorningBMumford & Sons|||TimshelBThe Killers|||SpacemanBPortishead|||RoadsB#Wham!|||Wake Me up Before You Go-GoBKendrick Lamar|||iBAriana Grande|||Love Me HarderBThe Offspring|||Self EsteemBFun.|||Carry OnBThe Black Keys|||Next GirlB,Eurythmics|||Sweet Dreams (Are Made of This)BSt. Vincent|||CruelB Iron & Wine|||Such Great HeightsBRegina Spektor|||FidelityB(Florence + The Machine|||Never Let Me GoB#The Ting Tings|||That's Not My NameBPearl Jam|||BlackBNeil Young|||Old ManB"Queen|||Another One Bites The DustBKnife Party|||BonfireB'Jimi Hendrix|||All Along The WatchtowerBKasabian|||UnderdogBSting|||Fields Of GoldBElla Henderson|||GhostBBlack Sabbath|||Iron ManB!Earth, Wind & Fire|||Let's GrooveBSheryl Crow|||All I Wanna DoB*Bob Marley & The Wailers|||Redemption SongBOneRepublic|||Good LifeBBirdy|||People Help The PeopleBYeah Yeah Yeahs|||ZeroBDisclosure|||You & MeB*Green Day|||Wake Me Up When September EndsBCat Stevens|||Wild WorldBLady Gaga|||PaparazziBColdplay|||Violet HillBMr. Mister|||Broken WingsBPitbull|||International LoveB"Duran Duran|||Hungry Like The WolfB-Enrique Iglesias|||Bailando - Spanish VersionBAerosmith|||CrazyBU2|||Sunday Bloody SundayB#The Rolling Stones|||Paint It BlackB(Fugees|||Killing Me Softly with His SongBBlondie|||One Way or AnotherBBastille|||OverjoyedBTears For Fears|||ShoutBAdele|||Make You Feel My LoveBOlly Murs|||TroublemakerBMumford & Sons|||I Gave You AllBDrake|||Best I Ever HadBKaty Perry|||This Is How We DoBChildish Gambino|||HeartbeatBSalt-N-Pepa|||Push ItBThe Killers|||Read My MindB0Rudimental|||Waiting All Night (feat. Ella Eyre)B!Jefferson Airplane|||White RabbitBIggy Azalea|||WorkBHot Chip|||Ready For The FloorBDuran Duran|||Ordinary WorldB Fleet Foxes|||Helplessness BluesBDaniel Powter|||Bad DayBR.E.M.|||Man On The MoonB#The Rolling Stones|||Under My ThumbB!Arctic Monkeys|||I Wanna Be YoursBEd Sheeran|||Kiss MeBSting|||Englishman In New YorkBWolfmother|||WomanBKe$ha|||Take It OffB!Justin Timberlake|||Tunnel VisionBTears For Fears|||Mad WorldBKenny Loggins|||Danger ZoneBEMF|||UnbelievableBDolly Parton|||JoleneBMKTO|||ClassicB The Rolling Stones|||Wild HorsesBRihanna|||Man DownBKanye West|||Good LifeBOutKast|||RosesBThe Fray|||You Found MeB*Bob Marley & The Wailers|||Buffalo SoldierBEminem|||MockingbirdB"Red Hot Chili Peppers|||By The WayB+The Mamas & The Papas|||California Dreamin'BTLC|||WaterfallsBJohnny Cash|||Personal JesusBLinkin Park|||One Step CloserBQueen|||Somebody To LoveBArctic Monkeys|||Snap Out Of ItBMuse|||Feeling GoodBMadonna|||Like A VirginBKelis|||MilkshakeBThe Kinks|||You Really Got MeBThe xx|||TryB)Bob Dylan|||The Times They Are A-Changin'B#Stevie Wonder|||For Once In My LifeBStevie Wonder|||Higher GroundBVanilla Ice|||Ice Ice BabyBThe Script|||SuperheroesBKe$ha|||BlowBOneRepublic|||SecretsBThe Ronettes|||Be My BabyBEric Clapton|||CocaineBRihanna|||DisturbiaB!Beyoncé|||Best Thing I Never HadBElvis Presley|||Hound DogBElvis Presley|||Blue ChristmasB!Arctic Monkeys|||One For The RoadBNirvana|||About A GirlBFoo Fighters|||WheelsBSantana|||SmoothB!George Michael|||Careless WhisperBAlicia Keys|||No OneBFoo Fighters|||BreakoutBThe Cranberries|||DreamsBJohn Mayer|||WildfireBDemi Lovato|||Heart AttackBMaroon 5|||DaylightB'Jimi Hendrix|||All Along the WatchtowerBBon Jovi|||RunawayB-Stealers Wheel|||Stuck In The Middle With YouBTaio Cruz|||HangoverBZZ Top|||La GrangeBJohnny Cash|||Ring of FireBSum 41|||Fat LipBEvanescence|||My ImmortalBLinkin Park|||What I've DoneB,Dropkick Murphys|||I'm Shipping Up To BostonBCake|||I Will SurviveBFirst Aid Kit|||EmmylouB(Queen|||Under Pressure - Remastered 2011BMaroon 5|||MiseryBDido|||Thank YouBFoxygen|||San FranciscoBRadiohead|||LuckyBWShakira|||Waka Waka (This Time for Africa) (The Official 2010 FIFA World Cup (TM) Song)BBeyoncé|||Sweet DreamsBAlabama Shakes|||I Found YouBBat For Lashes|||LauraBJustice|||GenesisBThe Turtles|||Happy TogetherBGinuwine|||PonyBEminem|||No LoveBJimi Hendrix|||Hey JoeBBritney Spears|||ToxicBwill.i.am|||This Is LoveBGreen Day|||21 GunsBMarvin Gaye|||What's Going OnBLady Gaga|||Do What U WantBLady Antebellum|||Need You NowBDiana Ross|||Upside DownBJungle|||TimeBNo Doubt|||Sunday MorningBRobbie Williams|||AngelsBFall Out Boy|||Dance, DanceB10cc|||I'm Not In LoveBSam Smith|||Lay Me DownB$Peter Bjorn And John|||Second ChanceBBastille|||IcarusBLana Del Rey|||Without YouBLimp Bizkit|||My WayB#The Supremes|||You Can't Hurry LoveB0Swedish House Mafia|||Save the World - Radio MixBThe Horrors|||Still LifeB"Kelly Clarkson|||Since U Been GoneBAdele|||LovesongBJohn Mayer|||GravityBFoo Fighters|||These DaysBColdplay|||Green EyesBChris Brown|||Look At Me NowBMisterWives|||ReflectionsBJack Johnson|||BreakdownBMike Will Made It|||23BWarpaint|||UndertowB&Otis Redding|||Try A Little TendernessBPhoenix|||GirlfriendBColdplay|||ShiverB$The National|||This Is The Last TimeB"The Calling|||Wherever You Will GoBMeredith Brooks|||BitchBTLC|||CreepBFoster The People|||Miss YouBDire Straits|||Romeo And JulietBThe La's|||There She GoesBZero 7|||DestinyBDido|||White FlagBLimp Bizkit|||Behind Blue EyesBSystem Of A Down|||B.Y.O.B.B0Omi|||Cheerleader - Felix Jaehn Remix Radio EditBBeck|||MorningBBell Biv DeVoe|||PoisonBGuns N' Roses|||PatienceBRobbie Williams|||FeelBLondon Grammar|||StrongBColdplay|||Warning SignB.Marvin Gaye|||I Heard It Through The GrapevineBN Sync|||Bye Bye ByeBThe xx|||ReunionBNorah Jones|||Turn Me OnBFoals|||InhalerBEdwin Starr|||WarB!Florence + The Machine|||SpectrumBPhantogram|||Fall In LoveBIyaz|||ReplayBEnrique Iglesias|||I Like ItBKings Of Leon|||Wait for MeBLana Del Rey|||West CoastBRyan Adams|||WonderwallB)The Righteous Brothers|||Unchained MelodyB)The Buggles|||Video Killed The Radio StarB"Katy Perry|||The One That Got AwayBPrince|||When Doves CryBFamily of the Year|||HeroBThe Prodigy|||FirestarterBLil Wayne|||Love MeB!Tears For Fears|||Head Over HeelsBRise Against|||SaviorBAerosmith|||Walk This WayBThe Black Keys|||The Only OneBThe Lumineers|||Slow It DownB"Phoenix|||Everything Is EverythingBDrake|||ForeverBThe 1975|||GirlsBBeyoncé|||CountdownBPhoenix|||FencesBEmeli Sandé|||HeavenBMadonna|||Material GirlBNine Inch Nails|||HurtBEminem|||SurvivalBThe National|||Sea of LoveBAriana Grande|||One Last TimeBQueen|||Under PressureBLil Wayne|||MirrorBKanye West|||AmazingBChris Brown|||ForeverBSara Bareilles|||Love SongBThe 1975|||SexBPantera|||WalkBChet Faker|||No DiggityBRoxette|||The LookBMadonna|||FrozenB7Yann Tiersen|||Comptine d'un autre été, l'après-midiBArctic Monkeys|||I Want It AllBIncubus|||Wish You Were HereBSkee-Lo|||I WishBM83|||OutroBJack Johnson|||ImagineBSnow Patrol|||Open Your EyesBOasis|||Live ForeverBThe xx|||MissingBTaylor Swift|||Love StoryBBryan Adams|||HeavenBHeart|||AloneBGoldfrapp|||Ooh La LaBCalvin Harris|||Let's GoBRihanna|||StayBYing Yang Twins|||Get LowB'Nina Simone|||My Baby Just Cares For MeBTimbaland|||ApologizeBMichael Bublé|||EverythingBOasis|||SupersonicB7The Smiths|||How Soon Is Now? (2008 Remastered Version)BThe Neighbourhood|||AfraidBJustin Timberlake|||That GirlBNicki Minaj|||AnacondaBKrewella|||AliveBNorah Jones|||SunriseBJimmy Eat World|||SweetnessBTanlines|||All Of MeBSara Bareilles|||GravityBImagine Dragons|||UnderdogBSmallpools|||DreamingBThe xx|||FantasyBChromatics|||LadyBCat Power|||Sea of LoveBRUN-DMC|||It's Like ThatBTaylor Swift|||22BBruce Springsteen|||The RiverB$Old Crow Medicine Show|||Wagon WheelBPhillip Phillips|||HomeB*Queens Of The Stone Age|||Go With The FlowBCoconut Records|||West CoastB#Phil Collins|||You Can't Hurry LoveBOne Direction|||Little ThingsBThe Fray|||Never Say NeverB!New Order|||Bizarre Love TriangleBRoxy Music|||More Than ThisBSylvan Esso|||CoffeeBDerek & The Dominos|||LaylaBJimi Hendrix|||Little WingBDisclosure|||VoicesBLed Zeppelin|||Whole Lotta LoveBDuran Duran|||Come UndoneBJames Arthur|||ImpossibleBDavid Gray|||BabylonBLil Wayne|||She WillBSade|||By Your SideBBeth Hirsch|||All I NeedBBen Howard|||The FearBThe National|||SorrowBEllie Goulding|||Your SongBJustin Bieber|||BabyB'Two Door Cinema Club|||This Is The LifeBB.o.B|||So GoodBCalvin Harris|||We Found LoveB Two Door Cinema Club|||Next YearBLily Allen|||Fuck YouB$James Vincent McMorrow|||Higher LoveB2The Beach Boys|||Good Vibrations - 2001 - RemasterBFlorence + The Machine|||HowlB#Fatboy Slim|||Right Here, Right NowBNick Jonas|||JealousBHot Chip|||I Feel BetterBM83|||IntroBDestiny's Child|||Say My NameBNas|||The MessageBKings Of Leon|||Sex On FireBMiami Horror|||SometimesBDanzig|||MotherBLed Zeppelin|||Immigrant SongBPlacebo|||Running Up That HillBColdplay|||TalkBBastille|||OblivionBWoodkid|||IronBFaith No More|||EasyBBeyoncé|||XOBCommodores|||EasyBJay Sean|||DownBMassive Attack|||AngelBOneRepublic|||ApologizeBThe Mowgli's|||San FranciscoBTycho|||AwakeBKris Kross|||JumpBGary Clark Jr.|||Bright LightsBBon Iver|||TeamBSum 41|||In Too DeepB&Queen|||Crazy Little Thing Called LoveBJAY Z|||New DayB!Everything But The Girl|||MissingBCrystal Fighters|||You & IBMadeon|||The CityBMumford & Sons|||BelieveBMadonna|||Express YourselfBSimon & Garfunkel|||AmericaBJAY Z|||Welcome To The JungleBWalk the Moon|||TightropeBYears & Years|||KingBKid Ink|||Show MeBCHVRCHES|||LiesB&Charlene Soraia|||Wherever You Will GoBGloria Gaynor|||I Will SurviveBRAC|||Let GoBThe Cars|||DriveBAlicia Keys|||Fallin'BJoni Mitchell|||A Case Of YouBMichael Kiwanuka|||Home AgainBThe xx|||Our SongB Ray Charles|||Georgia On My MindBJason Derulo|||In My HeadBColdplay|||What IfBKodaline|||High HopesBSummer Heart|||I Wanna GoB+Bruce Hornsby and the Range|||The Way It IsBEd Sheeran|||OneBRobbie Williams|||CandyBMiles Davis|||So WhatBKeane|||This Is The Last TimeBBruno Mars|||Count On MeBThe Byrds|||Mr. Tambourine ManBMuse|||Stockholm SyndromeBNe-Yo|||CloserBYouth Lagoon|||17B)Creedence Clearwater Revival|||Proud MaryBEminem|||The Way I AmBColdplay|||AmsterdamBFleetwood Mac|||LandslideBFleetwood Mac|||SongbirdB*The Asteroids Galaxy Tour|||The Golden AgeB#Lily Allen|||Somewhere Only We KnowBBest Coast|||BoyfriendBEd Sheeran|||PhotographBTaio Cruz|||HigherBJack Johnson|||Do You RememberBFoster The People|||Best FriendBLynyrd Skynyrd|||Simple ManB Amy Macdonald|||This Is The LifeB(The Black Keys|||Never Gonna Give You UpBMapei|||Don't WaitBZhu|||FadedB!Paul McCartney|||Live And Let DieBDrake|||OverBEve 6|||Inside OutBMariah Carey|||HeroBLil Wayne|||LollipopBCheap Trick|||SurrenderBBob Dylan|||Mr. Tambourine ManBSum 41|||PiecesBNirvana|||DumbBSam Cooke|||Wonderful WorldBLaura Marling|||GhostsBKaty Perry|||BirthdayBThe Naked And Famous|||No WayBNo Doubt|||It's My LifeBChristina Aguilera|||BeautifulBAriana Grande|||The WayBPearl Jam|||Better ManBMumford & Sons|||ReminderBNickelback|||Far AwayBKendrick Lamar|||RealBLenny Kravitz|||American WomanBSilverchair|||TomorrowBBob Dylan|||HurricaneBNena|||99 Red BalloonsBMadeon|||IcarusBBen E. King|||Stand By MeBCommon|||The LightBRöyksopp|||Do It AgainBMichael Jackson|||Human NatureBWill Smith|||MiamiBThe Who|||Behind Blue EyesBBananarama|||VenusBLed Zeppelin|||KashmirBChildish Gambino|||BonfireB'Aretha Franklin|||I Say A Little PrayerBMadeon|||FinaleBThe Drums|||MoneyBJohn Mayer|||XOBDrowning Pool|||BodiesBJustin Timberlake|||My LoveBKanye West|||Good MorningB!The Black Crowes|||Hard To HandleBDaughter|||HumanBIncubus|||Love HurtsBMr Little Jeans|||The SuburbsBVan Morrison|||Crazy LoveBU2|||All I Want Is YouBThe National|||RunawayBThe Monkees|||I'm A BelieverB'The Velvet Underground|||Sunday MorningBShaggy|||AngelB+The Animals|||Don't Let Me Be MisunderstoodBSnow Patrol|||RunBPendulum|||WitchcraftBThe Who|||Who Are YouBIcona Pop|||I Love ItB6Frank Sinatra|||Let It Snow! Let It Snow! Let It Snow!BHardwell|||SpacemanBThe Roots|||You Got MeBJosé González|||TeardropBTalk Talk|||It's My LifeBJoshua Radin|||WinterBStevie Wonder|||I WishBLana Del Rey|||Blue VelvetBBing Crosby|||White ChristmasB Guns N' Roses|||Live And Let DieBJennifer Paige|||CrushBAretha Franklin|||ThinkBGreen Day|||SheBVampire Weekend|||RunBEminem|||SupermanBJake Bugg|||BrokenBBombay Bicycle Club|||LunaBLimp Bizkit|||FaithBThe 1975|||The CityBThe Cure|||LovesongBLionel Richie|||HelloBMarilyn Manson|||Tainted LoveBHans Zimmer|||TimeBFriendly Fires|||ParisBCrystal Fighters|||FollowB+Totally Enormous Extinct Dinosaurs|||GardenBBroods|||BridgesBReal Estate|||EasyBMS MR|||BonesBThe Lemonheads|||Mrs. RobinsonBThe Pharcyde|||Runnin'BKings Of Leon|||ManhattanBMatchbox Twenty|||PushBJAY Z|||OceansBMatt and Kim|||Let's GoBKakkmaddafakka|||RestlessBDaughter|||MedicineBCaribou|||Our LoveBWilco|||You And IBSub Focus|||Tidal WaveB!Frankie Goes To Hollywood|||RelaxBSavages|||She WillBKimbra|||Settle DownBRay LaMontagne|||JoleneBAlphaville|||Forever YoungBWilson Phillips|||Hold OnBZedd|||SpectrumBSarah McLachlan|||AngelBIngrid Michaelson|||You and IBLisa Loeb & Nine Stories|||StayBLocal Natives|||You & IBJack Johnson|||AngelBRadiohead|||BonesBLady Antebellum|||DowntownBOne Direction|||One ThingBRegina Spektor|||HeroBLinkin Park|||RunawayBLeonard Cohen|||SuzanneBLimp Bizkit|||My GenerationBThe Strokes|||All The TimeBCreed|||HigherBMayer Hawthorne|||The WalkBClean Bandit|||Real LoveBBeastie Boys|||GirlsBDevendra Banhart|||BabyBNico|||These DaysBThe Rolling Stones|||Miss YouBU2|||VertigoBThe Kinks|||LolaBSelena Gomez|||Slow DownBNirvana|||Stay AwayBThe Prodigy|||Stand UpBThe Weeknd|||GoneBBusta Rhymes|||Thank YouBJack Johnson|||I Got YouB50 Cent|||Ayo TechnologyBGwen Stefani|||CoolBLifehouse|||EverythingBDelta Spirit|||CaliforniaBThe Kooks|||Shine OnBMadonna|||HolidayBGold Panda|||YouB8The Presidents Of The United States Of America|||PeachesBP!nk|||SoberBGrimes|||GoBDeerhunter|||HelicopterBFairground Attraction|||PerfectBPrince|||1999BThe National|||DemonsBKings Of Leon|||BirthdayBU2|||DesireBDarius Rucker|||Wagon WheelBSteely Dan|||Do It AgainBKings Of Leon|||I Want YouBBeach House|||WildBEminem|||When I'm GoneBJack Johnson|||Upside DownBSara Bareilles|||Winter SongBPhil Collins|||One More NightBLifehouse|||You And MeBDrake|||Too MuchBPJ Harvey|||Down By The WaterBJackson 5|||I'll Be ThereBYazoo|||Only YouBLinkin Park|||With YouBFun.|||All AloneBTaylor Swift|||RedBSugar Ray|||FlyBBen Howard|||PromiseB&Lauryn Hill|||Everything Is EverythingBParamore|||NowBParamore|||MonsterBBirdy|||1901BNext|||Too CloseBThe Doors|||The EndBDaughter|||HomeBBeyoncé|||I Miss YouB'Queens Of The Stone Age|||Little SisterBIncubus|||WarningBUsher|||I Don't MindBThe Middle East|||BloodBRobin S|||Show Me LoveBEmpire Of The Sun|||DNABKevin Lyttle|||Turn Me OnBJohn Mayer|||ClarityBJames Blake|||A Case Of YouBMuse|||SurvivalBLily Allen|||SmileBJessie Ware|||RunningBBirdy|||ShelterBPhoenix|||CountdownBBlondie|||RaptureBPortishead|||Machine GunBRilo Kiley|||Silver LiningBAvicii|||All You Need Is LoveBDavid Guetta|||SunshineBRY X|||BerlinBCut Copy|||Need You NowBAvenged Sevenfold|||NightmareBMariah Carey|||Without YouBDrake|||EnergyBAvicii|||Lay Me DownB3Creedence Clearwater Revival|||I Put A Spell On YouBBiffy Clyro|||MountainsBStyx|||RenegadeBFoals|||Bad HabitBChet Baker|||My Funny ValentineBStone Temple Pilots|||CreepBMajor Lazer|||Get FreeB$The Head And The Heart|||Winter SongBLondon Grammar|||NightcallBJohn Waite|||Missing YouBClassixx|||Holding OnBDavid Gray|||Sail AwayBNickelback|||RockstarBMadonna|||MusicBNickelback|||PhotographBThe Rapture|||Sail AwayBThe Corrs|||BreathlessBMr. Probz|||WavesBSeal|||CrazyBThe 1975|||Settle DownBCalvin Harris|||IronBMIKA|||Happy EndingB0Michael Bublé|||All I Want For Christmas Is YouBRöyksopp & Robyn|||Do It AgainBEtta James|||At LastBDaft Punk|||FallBGirls|||Lust For LifeBFoals|||MiamiBRihanna|||SkinBThe Rolling Stones|||HappyBRob Base|||It Takes TwoBThe Script|||NothingBFrou Frou|||Let GoB"Daryl Hall & John Oates|||ManeaterB"Thirty Seconds To Mars|||HurricaneBEd Sheeran|||Wake Me UpBKings Of Convenience|||HomesickBSonic Youth|||SuperstarBYazoo|||Don't GoBJAY Z|||HeavenBBeck|||WaveBCracker|||LowBJAY Z|||RenegadeBLeonard Cohen|||HallelujahB7Michael Bublé|||Have Yourself A Merry Little ChristmasBMadonna|||BorderlineB(Red Hot Chili Peppers|||Around The WorldBKanye West|||ParanoidB"Rage Against The Machine|||Wake UpBTycho|||DaydreamB"Toad The Wet Sprocket|||All I WantBKanye West|||Hey MamaBDillon Francis|||Get LowBMichael Bublé|||Feeling GoodBBeyoncé|||PartyBEarth, Wind & Fire|||FantasyBThe Neighbourhood|||Let It GoBFall Out Boy|||Alone TogetherB1Major Lazer, Amber of Dirty Projectors|||Get FreeBLondon Grammar|||Hey NowBKanye West|||GoneBRadiohead|||YouBEminem|||CriminalBJungle|||JuliaBAlesso|||CoolBThe Black Keys|||These DaysBAkon|||BeautifulB!Bob Marley & The Wailers|||ExodusBBloc Party|||SignsBFleetwood Mac|||GypsyBLCD Soundsystem|||HomeBThe Lumineers|||Morning SongBColdplay|||YesBNelly Furtado|||ManeaterBThe Guess Who|||American WomanBK-Ci & JoJo|||All My LifeBJason Mraz|||ButterflyBKings Of Leon|||The EndBColdplay|||DaylightBThe Bravery|||BelieveBJames Bay|||Let It GoBMuse|||Follow MeBBen Howard|||DiamondsBSpoon|||Inside OutBDaughter|||LoveBGeorge Michael|||FaithBTaylor Swift|||Our SongBGoldfinger|||99 Red BalloonsBMichael Bublé|||HomeBAtmosphere|||SunshineBJason Derulo|||BreathingBAtmosphere|||YesterdayBNicki Minaj|||FlyBDaft Punk|||OvertureBKula Shaker|||HushBCHIC|||Good TimesBT.I.|||What You KnowBGood Charlotte|||The RiverBFoxes|||YouthBSeether|||RemedyBDave Matthews Band|||You & MeBHaim|||Let Me GoBBest Coast|||Crazy For YouBBirdy|||Young BloodBCollective Soul|||ShineBWolfmother|||VagabondBDaft Punk|||FinaleB)Shaun Reynolds feat. Laura Pringle|||StayBGarbage|||Push ItB$The Decemberists|||Down By The WaterBWild Beasts|||WanderlustBThe Weeknd|||ValerieBShakira|||EmpireB.Michael Bublé|||Santa Claus Is Coming To TownB#KC & The Sunshine Band|||Give It UpBMichael Bublé|||Santa BabyBParamore|||HallelujahBSting|||FragileBSpice Girls|||StopBVan Halen|||JumpBJeff Buckley|||GraceBLady Gaga|||GypsyBJack Johnson|||Times Like TheseBLily Allen|||The FearBDaughter|||StillBBruce Springsteen|||I'm On FireBPoolside|||Harvest MoonBEagles|||Take It EasyBPnau|||EmbraceBDave Brubeck|||Take FiveB Florence + The Machine|||FallingBPassion Pit|||DreamsBJoshua Radin|||Only YouBAce of Base|||Beautiful LifeBDave Matthews Band|||SatelliteBFastball|||The WayBBeck|||GirlB!Pet Shop Boys|||Always On My MindB#She & Him|||Baby, It's Cold OutsideBP!nk|||True LoveBKanye West|||ChampionBThe Black Eyed Peas|||Shut UpBThe Box Tops|||The LetterBFall Out Boy|||Beat ItB+Michael Bublé|||I'll Be Home For ChristmasBJohn Mayer|||CrossroadsBAlpine|||GasolineBEd Sheeran|||RunawayBKings Of Leon|||CrawlBWhen In Rome|||The PromiseBwill.i.am|||Bang BangBBryan Adams|||When You're GoneBKings Of Leon|||17BTyga|||FadedBJake Bugg|||SlideBWhitesnake|||Is This LoveBThe Primitives|||CrashBWoodkid|||I Love YouBWashed Out|||EchoesBBruno Mars|||Show MeB)Jackson 5|||Santa Claus Is Coming To TownBRazorlight|||AmericaBThe Drums|||Best FriendBAnnie Lennox|||WhyBJerry Douglas|||The BoxerBThe Killers|||Romeo And JulietBKanye West|||I WonderBSeether|||Careless WhisperBMilow|||Ayo TechnologyBLighthouse Family|||HighBMichael Jackson|||ScreamBMadonna|||SorryBJay Sean|||Do You RememberBIncubus|||DigBThe Weeknd|||OutsideB
311|||DownBBritney Spears|||EverytimeBRadiohead|||Thinking About YouBAdele|||Crazy For YouBBombay Bicycle Club|||FeelBCaribou|||SilverBNew Order|||RegretBLenny Kravitz|||AgainBSavages|||Shut UpBGang Starr|||WorkBJoey Bada$$|||WavesBDaughtry|||Over YouB!Armin van Buuren|||Beautiful LifeBIggy Pop|||CandyBPaolo Nutini|||CandyBThe Vines|||Get FreeBtUnE-yArDs|||GangstaBFrank Sinatra|||My WayBJustin Bieber|||BoyfriendBLady Gaga|||You And IBWiz Khalifa|||When I'm GoneB#The Pussycat Dolls|||When I Grow UpBMarilyn Manson|||Personal JesusBJoni Mitchell|||RiverBThe Walkmen|||HeavenBPhil Collins|||Take Me HomeBPetula Clark|||DowntownBSystem Of A Down|||HypnotizeBJoni Mitchell|||CaliforniaBShinedown|||Second ChanceBDamien Rice|||I RememberBMariah Carey|||FantasyBEd Sheeran|||The CityBGabrielle|||DreamsBEllie Goulding|||Only YouBPanama|||AlwaysBParov Stelar|||All NightBChildish Gambino|||SoberBBeach House|||Take CareBJimi Hendrix|||FireBLady Gaga|||VenusBYellowcard|||Only OneBGotye|||I Feel BetterBKodaline|||One DayBSnow Patrol|||ChocolateB$The Dave Brubeck Quartet|||Take FiveBThe Killers|||BonesBBabylon Zoo|||SpacemanBNicki Minaj|||OnlyBBeck|||Say GoodbyeBFoals|||EverytimeBIcona Pop|||GirlfriendBSamantha Barks|||On My OwnBDisclosure|||Second ChanceBParamore|||FencesBDaft Punk|||NocturneB!The Smashing Pumpkins|||LandslideBRa Ra Riot|||Dance With MeBDaft Punk|||PhoenixBParamore|||PressureBTravis|||SingBBobby Helms|||Jingle Bell RockBOn An On|||GhostsBThe Weeknd|||The FallBThe Drums|||Down By The WaterBKrewella|||Come & Get ItB/TNGHT (Hudson Mohawke x Lunice)|||Higher GroundBMuse|||InvincibleBArmin van Buuren|||AloneBAlexander|||TruthBJamie xx|||GirlBJohnny Cash|||OneBLed Zeppelin|||HeartbreakerBBridgit Mendler|||Ready or NotB0Darlene Love|||Christmas (Baby Please Come Home)BYeah Yeah Yeahs|||RunawayBSimian Mobile Disco|||I BelieveBChris Brown|||With YouBCaribou|||SunBSavage Garden|||I Want YouBNine Inch Nails|||OnlyBKaty Perry|||Thinking Of YouBAkon|||AngelBChase & Status|||Count On MeBJunip|||AlwaysBDaft Punk|||Oh YeahBGrimes|||SkinBDaft Punk|||AliveBJoshua Radin|||CloserBChromatics|||I'm On FireBKings Of Leon|||MaryBDave Matthews Band|||CrushBBen Howard|||EverythingBStaind|||So Far AwayBMagic Man|||ParisBGotye|||Save MeBBleachers|||RollercoasterBSBTRKT|||HigherBJames Taylor|||MexicoBAvril Lavigne|||GirlfriendBMat Kearney|||Hey MamaBThe xx|||TogetherBColdplay|||LowBChristophe Beck|||WolvesBblink-182|||AlwaysB#Jamie Cullum|||Don't Stop The MusicBMatt Corby|||BrotherBFoals|||MoonBBanks|||DrowningB"John Coltrane|||My Favorite ThingsBChristophe Beck|||EpilogueB0OMI|||Cheerleader - Felix Jaehn Remix Radio EditB Ingrid Michaelson|||The Way I AmBGuster|||SatelliteBMuse|||AnimalsBABBA|||S.O.S.BMuse|||Save MeBAimee Mann|||Save MeBBreathe Carolina|||BlackoutBDaughtry|||HomeBBen Harper|||Walk AwayBBeck|||The Golden AgeBdeadmau5|||I RememberBSimply Red|||StarsBYouth Lagoon|||DaydreamBMartin Solveig|||HelloB!Sparklehorse|||Wish You Were HereBPat Benatar|||HeartbreakerBFaithless|||InsomniaB)Sixpence None The Richer|||There She GoesB-DJ Jazzy Jeff & The Fresh Prince|||SummertimeBDave Matthews Band|||Too MuchBblink-182|||DownBBob Dylan|||I Want YouBShinedown|||Call MeBblink-182|||CarouselBCrossfade|||ColdBIncubus|||I Miss YouBLykke Li|||TonightBUncle Kracker|||Follow MeBMadonna|||SecretB!Benjamin Francis Leftwich|||ShineBEd Sheeran|||The ManBAtoms For Peace|||DefaultBAerosmith|||AngelBAvenged Sevenfold|||AfterlifeBGlen Hansard|||Falling SlowlyBA Day To Remember|||All I WantB#Meaghan Smith|||Here Comes Your ManBThe Cure|||HighB!Stan Getz|||The Girl From IpanemaBThe Ting Tings|||HandsBTwo Door Cinema Club|||Wake UpB311|||Love SongB Dave Matthews Band|||Say GoodbyeBHurts|||Wonderful LifeBFall Out Boy|||I Don't CareBSmith Westerns|||WeekendBFever Ray|||When I Grow UpBTrain|||Marry MeB+Tom Petty And The Heartbreakers|||BreakdownBLadyhawke|||MagicBAir|||RememberB$Jr. Walker & The All Stars|||ShotgunBThe Smashing Pumpkins|||RocketB Stevie Ray Vaughan|||Little WingBGuano Apes|||Open Your EyesBSystem Of A Down|||SugarBAlabama Shakes|||HeartbreakerBGoldfinger|||SupermanBMarina and The Diamonds|||LiesB#Terence Trent D'Arby|||Wishing WellBSnoop Dogg|||SignsBBlack|||Wonderful LifeB*Daryl Hall & John Oates|||Jingle Bell RockBOtis Redding|||Hard To HandleBSpandau Ballet|||GoldBFirst Aid Kit|||BlueBPatsy Cline|||CrazyBJimmy Eat World|||PainBTwo Door Cinema Club|||SomedayBPearl Jam|||SirensBDaughter|||AmsterdamBFoals|||PreludeBLed Zeppelin|||Thank YouBBuddy Holly|||EverydayBWhite Hinterland|||IcarusBOne Direction|||You & IBMiguel|||All I Want Is YouBRhye|||WomanBBillie Holiday|||All of MeB?Creedence Clearwater Revival|||I Heard It Through The GrapevineBElvis Presley|||Blue MoonBRufus Wainwright|||HallelujahB2She & Him|||Have Yourself a Merry Little ChristmasBDr. Alban|||It's My LifeBNeil Young|||HelplessBThe Doors|||Light My FireBRihanna|||ComplicatedBThe Ronettes|||Sleigh RideBKasabian|||EmpireB"Bobby Womack|||California Dreamin'B!Imagine Dragons|||Round And RoundBblink-182|||AnthemBSkillet|||MonsterBThe Zutons|||ValerieBPeggy Lee|||FeverBYuck|||Get AwayBPink Floyd|||Wish You Were HereBCollege|||A Real HeroBAu Revoir Simone|||CrazyBLaura Mvula|||SheBJennifer Lopez|||Live It UpBMuse|||BlackoutBMuse|||InterludeBGoldroom|||EmbraceBAll Time Low|||WeightlessBSupertramp|||DreamerBEDillon Francis feat. Totally Enormous Extinct Dinosaurs|||Without YouBElvis Costello|||SheB%Grace Potter & The Nocturnals|||StarsBWeezer|||MemoriesBWashed Out|||Far AwayBSt. Lucia|||SeptemberBRay LaMontagne|||ShelterBBirdy|||White Winter HymnalBKool & The Gang|||CelebrationBThe Smashing Pumpkins|||LoveBHugo|||99 ProblemsBJess Glynne|||Hold My HandBLucenzo|||Danza KuduroBWild Nothing|||NocturneBKate Boy|||Northern LightsBMariah Carey|||HeartbreakerBTanlines|||BrothersBLudacris|||Stand UpBGarbage|||When I Grow UpB%Amy Winehouse|||The Girl From IpanemaBThe Smashing Pumpkins|||BodiesBBlondie|||MariaBTaylor Swift|||MineBKylie Minogue|||In Your EyesB2Grandmaster Flash & The Furious Five|||The MessageBMuse|||BlissBOasis|||Stand By MeBBasement Jaxx|||Never Say NeverBShawn Mullins|||LullabyBLMFAO|||YesBJimmy Eat World|||WorkBFranz Ferdinand|||Walk AwayBZendaya|||ReplayB6Frank Sinatra|||Have Yourself a Merry Little ChristmasBAmy Winehouse|||AddictedBJoni Mitchell|||All I WantBElla Fitzgerald|||Sleigh RideBNoosa|||Walk On ByBPriscilla Ahn|||DreamBJustin Timberlake|||Never AgainBGrouplove|||Let Me InBThe Smashing Pumpkins|||PerfectBAlpine|||HandsBDrake|||FancyBBombay Bicycle Club|||StillBHot Chip|||Night And DayBCarole King|||It's Too LateBDisclosure|||IntroBEminem|||On FireBThe Black Keys|||LiesBEartha Kitt|||Santa BabyBBilly Joel|||My LifeB#Bombay Bicycle Club|||What You WantBColbie Caillat|||TryBFun.|||StarsBMaroon 5|||SecretBThe Cranberries|||SalvationBIngrid Michaelson|||EverybodyBPatrick Watson|||LighthouseBElton John|||DanielBKanye West|||CelebrationB#Selena Gomez & The Scene|||Who SaysBNew Order|||TemptationBThe Kooks|||SwayBTV On The Radio|||RideB2Michael Bublé|||Christmas (Baby Please Come Home)BAlanis Morissette|||Thank YouBKorn|||BlindBLupe Fiasco|||SunshineBWashed Out|||You and IBJunip|||Without YouB'Tony Bennett|||The Way You Look TonightBEric Clapton|||PromisesBThe Black Crowes|||RemedyBFoo Fighters|||HomeBThe Beatles|||Hey JudeBR.E.M.|||DriveBBritney Spears|||StrongerBMaroon 5|||The SunBMatisyahu|||SunshineBNickelback|||SomedayBMariah Carey|||HoneyBTV On The Radio|||WinterBDes'ree|||LifeBMoby|||FlowerBM-Clan|||CarolinaBOrgy|||Blue MondayBLThe Presidents Of The United States Of America|||Video Killed The Radio StarB Justin Bieber|||Somebody To LoveBRodriguez|||I WonderBShe & Him|||Don't Look BackBWarrant|||HeavenB"Rage Against The Machine|||FreedomB"Nina Simone|||I Put A Spell On YouBBahamas|||All The TimeBSufjan Stevens|||Amazing GraceBDeerhunter|||SleepwalkingBMåns Zelmerlöw|||HeroesBEminem|||BeautifulBAlex Clare|||Up All NightB"Vince Guaraldi Trio|||GreensleevesBMat Kearney|||DownBDave Matthews Band|||MercyBEd Sheeran|||Take It BackBAmos Lee|||ColorsBFoo Fighters|||LowBParamore|||MiracleBBobby Hebb|||SunnyBOne Direction|||Heart AttackBGorillaz|||All AloneBLabrinth|||Express YourselfBRegina Spektor|||BetterBOzzy Osbourne|||DreamerBHugh Jackman|||SuddenlyBLeon Bridges|||Coming HomeB%The Whitest Boy Alive|||Don't Give UpBNero|||InnocenceBGeorge Michael|||OutsideBWillie Nelson|||The ScientistBMy Chemical Romance|||MamaBMary J. Blige|||Real LoveBThe Offspring|||All I WantBAWOLNATION|||All I NeedBN Sync|||GoneBNazareth|||Love HurtsBAne Brun|||Do You RememberBTina Turner|||Proud MaryBThe Strokes|||Happy EndingBDead by April|||Losing YouBSharon Van Etten|||Our LoveBMuse|||PreludeBThe Cure|||The WalkBMichael Bublé|||Silent NightBX Ambassadors|||JungleBJustin Bieber|||Love MeBChris Isaak|||Wicked GameBEnrique Iglesias|||LocoBMartin Garrix|||HelicopterBJimmy Eat World|||23B$Ariel Pink's Haunted Graffiti|||BabyBThe Prodigy|||ThunderB'Barry Louis Polisar|||All I Want Is You
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
j

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name25*
value_dtype0	
q

embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�0*
shared_name
embeddings
j
embeddings/Read/ReadVariableOpReadVariableOp
embeddings*
_output_shapes
:	�0*
dtype0
q

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�0*
shared_name
candidates
j
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes
:	�0*
dtype0
o
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameidentifiers
h
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes	
:�*
dtype0
r
serving_default_input_1Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
hash_tableConst_2
embeddings
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_signature_wrapper_440
�
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference__initializer_597
(
NoOpNoOp^StatefulPartitionedCall_1
�
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures*

0
	1

2*
* 
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 

	capture_1* 
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*
* 

$serving_default* 
JD
VARIABLE_VALUE
embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

0
	1

2*

0*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 
9
%	keras_api
&input_vocabulary
'lookup_table* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

embeddings*

0*
* 
* 
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 

	capture_1* 
* 
* 
R
;_initializer
<_create_resource
=_initialize
>_destroy_resource* 

0*
* 
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 

0*

0
1*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 

Ftrace_0* 

Gtrace_0* 

Htrace_0* 

0*
* 
* 
* 
* 
* 
* 
* 
 
I	capture_1
J	capture_2* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers/Read/ReadVariableOpcandidates/Read/ReadVariableOpembeddings/Read/ReadVariableOpConst_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference__traced_save_640
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameidentifiers
candidates
embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_restore_659ފ
�
8
__inference__creator_589
identity��
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name25*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
(__inference_sequential_layer_call_fn_193
string_lookup_input
unknown
	unknown_0	
	unknown_1:	�0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_555

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_549:	�0
identity��embedding/embedding_lookup�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_549string_lookup/Identity:output:0*
Tindices0	*1
_class'
%#loc:@embedding/embedding_lookup/549*'
_output_shapes
:���������0*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*1
_class'
%#loc:@embedding/embedding_lookup/549*'
_output_shapes
:���������0�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0}
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_542

inputs
unknown
	unknown_0	
	unknown_1:	�0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
B__inference_embedding_layer_call_and_return_conditional_losses_584

inputs	'
embedding_lookup_578:	�0
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_578inputs*
Tindices0	*'
_class
loc:@embedding_lookup/578*'
_output_shapes
:���������0*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_class
loc:@embedding_lookup/578*'
_output_shapes
:���������0}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������0Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_184

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_180:	�0
identity��!embedding/StatefulPartitionedCall�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_180*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_179y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_225

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_221:	�0
identity��!embedding/StatefulPartitionedCall�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_221*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_179y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
B__inference_embedding_layer_call_and_return_conditional_losses_179

inputs	'
embedding_lookup_173:	�0
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_173inputs*
Tindices0	*'
_class
loc:@embedding_lookup/173*'
_output_shapes
:���������0*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_class
loc:@embedding_lookup/173*'
_output_shapes
:���������0}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������0Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
)__inference_brute_force_layer_call_fn_457
queries
unknown
	unknown_0	
	unknown_1:	�0
	unknown_2:	�0
	unknown_3:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_brute_force_layer_call_and_return_conditional_losses_292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_292
queries
sequential_274
sequential_276	!
sequential_278:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_274sequential_276sequential_278*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_184u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
__inference__traced_save_640
file_prefix*
&savev2_identifiers_read_readvariableop)
%savev2_candidates_read_readvariableop)
%savev2_embeddings_read_readvariableop
savev2_const_3

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_identifiers_read_readvariableop%savev2_candidates_read_readvariableop%savev2_embeddings_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*4
_input_shapes#
!: :�:	�0:	�0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:%!

_output_shapes
:	�0:%!

_output_shapes
:	�0:

_output_shapes
: 
�	
�
)__inference_brute_force_layer_call_fn_379
input_1
unknown
	unknown_0	
	unknown_1:	�0
	unknown_2:	�0
	unknown_3:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_brute_force_layer_call_and_return_conditional_losses_347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�	
�
)__inference_brute_force_layer_call_fn_307
input_1
unknown
	unknown_0	
	unknown_1:	�0
	unknown_2:	�0
	unknown_3:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_brute_force_layer_call_and_return_conditional_losses_292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�	
�
!__inference_signature_wrapper_440
input_1
unknown
	unknown_0	
	unknown_1:	�0
	unknown_2:	�0
	unknown_3:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__wrapped_model_159o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_568

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_562:	�0
identity��embedding/embedding_lookup�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_562string_lookup/Identity:output:0*
Tindices0	*1
_class'
%#loc:@embedding/embedding_lookup/562*'
_output_shapes
:���������0*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*1
_class'
%#loc:@embedding/embedding_lookup/562*'
_output_shapes
:���������0�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0}
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_245
string_lookup_input
unknown
	unknown_0	
	unknown_1:	�0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_347
queries
sequential_329
sequential_331	!
sequential_333:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_329sequential_331sequential_333*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_225u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
__inference__traced_restore_659
file_prefix+
assignvariableop_identifiers:	�0
assignvariableop_1_candidates:	�00
assignvariableop_2_embeddings:	�0

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_4Identity_4:output:0*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference__initializer_5975
1key_value_init24_lookuptableimportv2_table_handle-
)key_value_init24_lookuptableimportv2_keys/
+key_value_init24_lookuptableimportv2_values	
identity��$key_value_init24/LookupTableImportV2�
$key_value_init24/LookupTableImportV2LookupTableImportV21key_value_init24_lookuptableimportv2_table_handle)key_value_init24_lookuptableimportv2_keys+key_value_init24_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init24/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :�:�2L
$key_value_init24/LookupTableImportV2$key_value_init24/LookupTableImportV2:!

_output_shapes	
:�:!

_output_shapes	
:�
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_267
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_263:	�0
identity��!embedding/StatefulPartitionedCall�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_263*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_179y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_421
input_1
sequential_403
sequential_405	!
sequential_407:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_403sequential_405sequential_407*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_225u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_497
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
)sequential_embedding_embedding_lookup_481:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�%sequential/embedding/embedding_lookup�6sequential/string_lookup/None_Lookup/LookupTableFindV2�
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
%sequential/embedding/embedding_lookupResourceGather)sequential_embedding_embedding_lookup_481*sequential/string_lookup/Identity:output:0*
Tindices0	*<
_class2
0.loc:@sequential/embedding/embedding_lookup/481*'
_output_shapes
:���������0*
dtype0�
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@sequential/embedding/embedding_lookup/481*'
_output_shapes
:���������0�
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
|
'__inference_embedding_layer_call_fn_575

inputs	
unknown:	�0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_520
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
)sequential_embedding_embedding_lookup_504:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�%sequential/embedding/embedding_lookup�6sequential/string_lookup/None_Lookup/LookupTableFindV2�
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
%sequential/embedding/embedding_lookupResourceGather)sequential_embedding_embedding_lookup_504*sequential/string_lookup/Identity:output:0*
Tindices0	*<
_class2
0.loc:@sequential/embedding/embedding_lookup/504*'
_output_shapes
:���������0*
dtype0�
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@sequential/embedding/embedding_lookup/504*'
_output_shapes
:���������0�
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
*
__inference__destroyer_602
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_256
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_252:	�0
identity��!embedding/StatefulPartitionedCall�+string_lookup/None_Lookup/LookupTableFindV2�
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_252*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_179y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0�
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�	
�
)__inference_brute_force_layer_call_fn_474
queries
unknown
	unknown_0	
	unknown_1:	�0
	unknown_2:	�0
	unknown_3:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_brute_force_layer_call_and_return_conditional_losses_347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
__inference__wrapped_model_159
input_1S
Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleT
Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value	H
5brute_force_sequential_embedding_embedding_lookup_143:	�0=
*brute_force_matmul_readvariableop_resource:	�0*
brute_force_gather_resource:	�
identity

identity_1��brute_force/Gather�!brute_force/MatMul/ReadVariableOp�1brute_force/sequential/embedding/embedding_lookup�Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2�
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleinput_1Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
-brute_force/sequential/string_lookup/IdentityIdentityKbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
1brute_force/sequential/embedding/embedding_lookupResourceGather5brute_force_sequential_embedding_embedding_lookup_1436brute_force/sequential/string_lookup/Identity:output:0*
Tindices0	*H
_class>
<:loc:@brute_force/sequential/embedding/embedding_lookup/143*'
_output_shapes
:���������0*
dtype0�
:brute_force/sequential/embedding/embedding_lookup/IdentityIdentity:brute_force/sequential/embedding/embedding_lookup:output:0*
T0*H
_class>
<:loc:@brute_force/sequential/embedding/embedding_lookup/143*'
_output_shapes
:���������0�
<brute_force/sequential/embedding/embedding_lookup/Identity_1IdentityCbrute_force/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������0�
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
brute_force/MatMulMatMulEbrute_force/sequential/embedding/embedding_lookup/Identity_1:output:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(V
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
�
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0o
brute_force/IdentityIdentitybrute_force/Gather:output:0*
T0*'
_output_shapes
:���������
j
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
n

Identity_1Identitybrute_force/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOp2^brute_force/sequential/embedding/embedding_lookupC^brute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2f
1brute_force/sequential/embedding/embedding_lookup1brute_force/sequential/embedding/embedding_lookup2�
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
D__inference_brute_force_layer_call_and_return_conditional_losses_400
input_1
sequential_382
sequential_384	!
sequential_386:	�01
matmul_readvariableop_resource:	�0
gather_resource:	�

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_382sequential_384sequential_386*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_184u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�0*
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_531

inputs
unknown
	unknown_0	
	unknown_1:	�0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: "�	L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input_1,
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
<
output_20
StatefulPartitionedCall:1���������
tensorflow/serving/predict:؂
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures"
_tf_keras_model
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
)__inference_brute_force_layer_call_fn_307
)__inference_brute_force_layer_call_fn_457
)__inference_brute_force_layer_call_fn_474
)__inference_brute_force_layer_call_fn_379�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
D__inference_brute_force_layer_call_and_return_conditional_losses_497
D__inference_brute_force_layer_call_and_return_conditional_losses_520
D__inference_brute_force_layer_call_and_return_conditional_losses_400
D__inference_brute_force_layer_call_and_return_conditional_losses_421�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
	capture_1B�
__inference__wrapped_model_159input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_sequential
:�2identifiers
:	�02
candidates
�2��
���
FullArgSpec1
args)�&
jself
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
$serving_default"
signature_map
:	�02
embeddings
5
0
	1

2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
)__inference_brute_force_layer_call_fn_307input_1"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
)__inference_brute_force_layer_call_fn_457queries"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
)__inference_brute_force_layer_call_fn_474queries"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
)__inference_brute_force_layer_call_fn_379input_1"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
D__inference_brute_force_layer_call_and_return_conditional_losses_497queries"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
D__inference_brute_force_layer_call_and_return_conditional_losses_520queries"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
D__inference_brute_force_layer_call_and_return_conditional_losses_400input_1"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
D__inference_brute_force_layer_call_and_return_conditional_losses_421input_1"�
���
FullArgSpec#
args�
jself
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
!J	
Const_2jtf.TrackableConstant
P
%	keras_api
&input_vocabulary
'lookup_table"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
3trace_0
4trace_1
5trace_2
6trace_32�
(__inference_sequential_layer_call_fn_193
(__inference_sequential_layer_call_fn_531
(__inference_sequential_layer_call_fn_542
(__inference_sequential_layer_call_fn_245�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3trace_0z4trace_1z5trace_2z6trace_3
�
7trace_0
8trace_1
9trace_2
:trace_32�
C__inference_sequential_layer_call_and_return_conditional_losses_555
C__inference_sequential_layer_call_and_return_conditional_losses_568
C__inference_sequential_layer_call_and_return_conditional_losses_256
C__inference_sequential_layer_call_and_return_conditional_losses_267�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0z8trace_1z9trace_2z:trace_3
�
	capture_1B�
!__inference_signature_wrapper_440input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
 "
trackable_list_wrapper
f
;_initializer
<_create_resource
=_initialize
>_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
'__inference_embedding_layer_call_fn_575�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
�
Etrace_02�
B__inference_embedding_layer_call_and_return_conditional_losses_584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
'
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
(__inference_sequential_layer_call_fn_193string_lookup_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
(__inference_sequential_layer_call_fn_531inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
(__inference_sequential_layer_call_fn_542inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
(__inference_sequential_layer_call_fn_245string_lookup_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
C__inference_sequential_layer_call_and_return_conditional_losses_555inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
C__inference_sequential_layer_call_and_return_conditional_losses_568inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
C__inference_sequential_layer_call_and_return_conditional_losses_256string_lookup_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
C__inference_sequential_layer_call_and_return_conditional_losses_267string_lookup_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
�
Ftrace_02�
__inference__creator_589�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zFtrace_0
�
Gtrace_02�
__inference__initializer_597�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zGtrace_0
�
Htrace_02�
__inference__destroyer_602�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zHtrace_0
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_embedding_layer_call_fn_575inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_embedding_layer_call_and_return_conditional_losses_584inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference__creator_589"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
I	capture_1
J	capture_2B�
__inference__initializer_597"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zI	capture_1zJ	capture_2
�B�
__inference__destroyer_602"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant4
__inference__creator_589�

� 
� "� 6
__inference__destroyer_602�

� 
� "� =
__inference__initializer_597'IJ�

� 
� "� �
__inference__wrapped_model_159�'
	,�)
"�
�
input_1���������
� "c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
�
D__inference_brute_force_layer_call_and_return_conditional_losses_400�'
	@�=
&�#
�
input_1���������

 
�

trainingp "K�H
A�>
�
0/0���������

�
0/1���������

� �
D__inference_brute_force_layer_call_and_return_conditional_losses_421�'
	@�=
&�#
�
input_1���������

 
�

trainingp"K�H
A�>
�
0/0���������

�
0/1���������

� �
D__inference_brute_force_layer_call_and_return_conditional_losses_497�'
	@�=
&�#
�
queries���������

 
�

trainingp "K�H
A�>
�
0/0���������

�
0/1���������

� �
D__inference_brute_force_layer_call_and_return_conditional_losses_520�'
	@�=
&�#
�
queries���������

 
�

trainingp"K�H
A�>
�
0/0���������

�
0/1���������

� �
)__inference_brute_force_layer_call_fn_307�'
	@�=
&�#
�
input_1���������

 
�

trainingp "=�:
�
0���������

�
1���������
�
)__inference_brute_force_layer_call_fn_379�'
	@�=
&�#
�
input_1���������

 
�

trainingp"=�:
�
0���������

�
1���������
�
)__inference_brute_force_layer_call_fn_457�'
	@�=
&�#
�
queries���������

 
�

trainingp "=�:
�
0���������

�
1���������
�
)__inference_brute_force_layer_call_fn_474�'
	@�=
&�#
�
queries���������

 
�

trainingp"=�:
�
0���������

�
1���������
�
B__inference_embedding_layer_call_and_return_conditional_losses_584W+�(
!�
�
inputs���������	
� "%�"
�
0���������0
� u
'__inference_embedding_layer_call_fn_575J+�(
!�
�
inputs���������	
� "����������0�
C__inference_sequential_layer_call_and_return_conditional_losses_256n'@�=
6�3
)�&
string_lookup_input���������
p 

 
� "%�"
�
0���������0
� �
C__inference_sequential_layer_call_and_return_conditional_losses_267n'@�=
6�3
)�&
string_lookup_input���������
p

 
� "%�"
�
0���������0
� �
C__inference_sequential_layer_call_and_return_conditional_losses_555a'3�0
)�&
�
inputs���������
p 

 
� "%�"
�
0���������0
� �
C__inference_sequential_layer_call_and_return_conditional_losses_568a'3�0
)�&
�
inputs���������
p

 
� "%�"
�
0���������0
� �
(__inference_sequential_layer_call_fn_193a'@�=
6�3
)�&
string_lookup_input���������
p 

 
� "����������0�
(__inference_sequential_layer_call_fn_245a'@�=
6�3
)�&
string_lookup_input���������
p

 
� "����������0�
(__inference_sequential_layer_call_fn_531T'3�0
)�&
�
inputs���������
p 

 
� "����������0�
(__inference_sequential_layer_call_fn_542T'3�0
)�&
�
inputs���������
p

 
� "����������0�
!__inference_signature_wrapper_440�'
	7�4
� 
-�*
(
input_1�
input_1���������"c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
