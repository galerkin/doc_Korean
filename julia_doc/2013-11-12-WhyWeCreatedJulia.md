왜 줄리아를 만들었나

authors:
    - <a href="http://github.com/JeffBezanson/">Jeff Bezanson</a>
    - <a href="http://karpinski.org/">Stefan Karpinski</a>
    - <a href="http://www.allthingshpc.org/">Viral Shah</a>
    - <a href="http://www-math.mit.edu/~edelman/">Alan Edelman</a>

trackbacks:
    - [Reddit]("http://www.reddit.com/r/programming/comments/pv3k9/why_we_created_julia_a_new_programming_language/")
    - [Hacker News]( "http://news.ycombinator.com/item?id=3606380")
    - [Lambda-the-Ultimate]( "http://lambda-the-ultimate.org/node/4452")
    - [Phoronix]( "http://www.phoronix.com/scan.php?page=news_item&px=MTA2ODg")
    - [The Endeavor (John D. Cook)]( "http://www.johndcook.com/blog/2012/02/22/julia-random-number-generation/")
    - [Walking Randomly]( "http://www.walkingrandomly.com/?p=87")
    - [Miguel Bazdresch]( "http://2pif.info/op/julia.html")
    - [Douglas Bates] ("http://dmbates.blogspot.in/2012/03/julia-version-of-multinomial-sampler_12.html")
    - [Vince Buffalo] ("http://vincebuffalo.org/2012/03/07/thoughts-on-julia.html")
    - [John Myles White] ("http://www.johnmyleswhite.com/notebook/2012/03/31/julia-i-love-you/")
    - [Shane Conway] ("http://www.statalgo.com/2012/03/24/statistics-with-julia/")
    - [Tim Salimans] ("http://timsalimans.com/gibbs-sampling-with-julia/")
    - [Twitter trackbacks]("http://topsy.com/julialang.org/")
    - [Russian translation]("http://habrahabr.ru/blogs/programming/138577/")
    - [Linux.org.ru blog post (Russian)]("http://www.linux.org.ru/news/opensource/7440863")
    - [Simplified Chinese translation]("http://sd.csdn.net/a/20120223/312315.html")
    - [Traditional Chinese translation] ("http://www.hellogcc.org/archives/666")
    - [Linuxfr.org blog post (French)] ("http://linuxfr.org/news/version-1-0-de-julia")

간단히 말해서, 우리가 욕심꾸러기이기 때문이다.

우리는 Matlab을 파워유저들이고, 거기에 몇몇은 Lisp 해커들이다. 또 파이썬및 Ruby 전문가가다, 또한, 여전히 Perl 해커이기도 하다. 수염이 나기 전부터 매쓰매티카를 사용해온 사람들도 있고, 여전히 수염이 없는 이도 있다. 멀쩡한 사람들보다 훨씬 많은 R 플롯을 만들어왔다. C는 무인도에 갈때 고를 한가지 프로그래밍 언어다.

우리는이 언어들을 사랑한다; 이들 언어들은  완벽하고 강하다. 직장에서 우리는 과학 컴퓨팅, 기계 학습, 데이터 마이닝, 대규모 병렬 및 분산 컴퓨팅 선형 대수학 연구 작업을 많이 할 - 각 언어들은 자신의 장점을 가지는 작업이 있고, 특정 작업에는 엉망이다. 각 언어는 이율 배반적이다.

우리는 욕심꾸러기다: 더 많은 걸 원한다.

우리는 자유로운 라이센스와 오픈 소스 언어를 원한다. 우리는 Ruby의 역동t과 C 의 속도 를 원한다. 우리는 언어 Lisp 같은 진짜 매크로에서 homoiconic(자료구조와 프로그램의 형태가 같은) 하지만 Matlab 과 같은 명백한 익숙한 수학적 표기 하고 싶다. 쉘과 같이 프로그램 잘 붙이면서, 좋은 Matlab 과 같이 선형 대수학 을위한 강력한,  Perl의 문자열 처리처럼  자연스러우며, R의 통계 처럼 쉽고, Python의 일반적인 프로그래밍용으로 사용할 수있는 무언가를 갖고 싶다. 배우는 것은 엄청 쉬우면서, 여전히 가장 심각한 해커 행복하게하는 무엇이. 우리는 대화형으로 쓰고 동시에 그것을 컴파일하고 싶다.

(그것을 C 만큼 빨라야 한다는 말 했나요?)
요구하면, 수 킬로바이트의 정형된 Java와 XML없이 Hadoop의 분산 컴퓨팅을 지원할 무언가를 원한다; 
While we're being demanding, we want something that provides the distributed power of Hadoop — without the kilobytes of boilerplate Java and XML;
without being forced to sift through gigabytes of log files on hundreds of machines to find our bugs.
We want the power without the layers of impenetrable complexity.
We want to write simple scalar loops that compile down to tight machine code using just the registers on a single CPU.
We want to write `A*B` and launch a thousand computations on a thousand machines, calculating a vast matrix product together.

We never want to mention types when we don't feel like it.
But when we need polymorphic functions, we want to use generic programming to write an algorithm just once and apply it to an infinite lattice of types;
we want to use multiple dispatch to efficiently pick the best method for all of a function's arguments, from dozens of method definitions, providing common functionality across drastically different types.
Despite all this power, we want the language to be simple and clean.

All this doesn't seem like too much to ask for, does it?

Even though we recognize that we are inexcusably greedy, we still want to have it all.
About two and a half years ago, we set out to create the language of our greed.
It's not complete, but it's time for a 1.0 release — the language we've created is called [Julia](/).
It already delivers on 90% of our ungracious demands, and now it needs the ungracious demands of others to shape it further.
So, if you are also a greedy, unreasonable, demanding programmer, we want you to give it a try.
