#Fast Numeric Computation in Julia
---
layout: post
title:  Fast Numeric Computation in Julia
author: <a href="http://dahua.me">Dahua Lin</a>
---
translator: [Jaesung Eom](https://twitter.com/JaesungEom), Nov 8th, 2013

매일 수치 해석 문제를 다루며, 항상 큰 데이터를 다루는  엄청나게 빠른 실행 코드를 작성할 수 있도록하면서 우아한 인터페이스를 제공하는 언어를 꿈꿔왔다. 줄리아는 꿈을 현실로 이뤄주는 언어다.
줄리아와 함께, 여러분 문제에 보다 초점을 맞출 수 있으면서도 코드를 깨끗하게 유지하고, 더 중요한 것은, 성능이 중요한 경우에도 C 나 포트란과 같은 저수준의 언어로 다이빙하지 않고도 빠르게 작동하는  코드를 작성할수있다.

하지만, 그렇다고 이러한 가능성이 거저 얻어지진 않는다. 당신의 코드를 빠르게 하려면, 성능을 염두에 두고 이 문서에서 소개하는 일반적인 모범 사례 지침을 따라야한다. 이제 수치 계산을위한 효율적인 코드를 작성하는 내 경험을 공유한다.

## 첫째, 제대로 만들라 (First, make it correct)

다른 프로그램언어와 같이, 여러분의 알고리즘을 구현할 때, 가장 중요한 목표는 *제대로 작동하게* 하는 것이다.
작동하지 않는 알고리즘은 아무리 빨리 실행되도 쓸모없다. 필요한 경우 나중에 코드를 언제나  최적화 할 수 있다.
문제에 대한 다른 접근 방법이 있을 때, **점근적으로 더 효율적인** 방법을 선택해야합니다.
예를 들어,  최적화 안된 퀵소트 구현도 쉽사리 적당히 큰 크기 배열을 정렬 할 때 신중하게 최적화된 버블소트보다 빠를 수 있다.
각 알고리즘 특수성을 감안하더라도,대부분 신중하게 구현하고, 일반적인 성능 지침을 준수하는 것은 성능에 큰 차이를 만들 수 있습니다 - 이글의 나머지에서 여기 초점을 맞춰보자.

## 디벡터라이제이션(비벡터화) 표현식 (Devectorize expressions)

매트랩<sup>®</sup>나 파이썬 같은 고수준 언어의 사용자들은 성능을 위해 종종 그들의 코드를 최대한 **벡터라이제이션**하라는 충고를 듣는다. 왜냐면, 그 언어들에서는 루프가 느리기 때문이다. 줄리아에서는, 반대로, 루프가 C로 작성하는 만큼 빠르게 실행되고, 속도를 위해 벡터화에 의존하지 않아도 된다. 사실 벡터화 표현식을 루프로 바꾸는게- 우린 **디벡터라이제이션**이라 부른다- 종종 더 높은 성능을 보여준다

다음 경우를 보자:


    r = exp(-abs(x-y))

아주 간단한 식이다.
무대뒤에서는, 하지만, 이 식의 결과를 얻으려면, 여러 단계와 임시 배열들이 쓰인다. 다음의 일련의 임시 배열은 위의 수식을 계산하기 위해 필요한 것들이다:

    n = length(x)

    tmp1 = Array(Float64, n)
    for i = 1:n
        tmp1[i] = x[i]-y[i]
    end

    tmp2 = Array(Float64, n)
    for i = 1:n
        tmp2[i] = abs(tmp1[i])
    end

    tmp3 = Array(Float64, n)
    for i = 1:n
        tmp3[i] = -tmp2[i]
    end

    r = Array(Float64, n)
    for i = 1:n
        r[i] = exp(tmp3[i])
    end

여기서 계산을 마치기 위해, 세 개의 임시 배열을 생성하고 네 단계를 거치는 걸 보았다.
이는 상당한 오버헤드를 유발한다:

* 임시배열에 메모리 할당 할 비용;
* 가비지 콜렉션동안 이 임시 배열 메모리 회수비용;
* 메모리를 훝어가는 비용-대게 패스회수가 적을 수록 효율이 높다.

실제 이러한 오버해드는 작업을 두세배 느리게한다. 최적의 성능을 얻으려면 다음과 같이 **디벡터라이제이션**한다:

    r = similar(x) 
    for i = 1:length(x)
        r[i] = exp(-abs(x[i]-y[i]))
    end

이 버젼은 임시 배열을 사용하지 않고 한번에 계산을 완료한다. 게다가, 만약 `r` 이 미리 할당되어 있다면, `r`을 생성하는 과정을 생략할 수 있다. [*Devectorize.jl*](https://github.com/lindahua/Devectorize.jl) 패키지는 ``@devec`` 매크로를 제공하고, 자동으로 벡터화 표현식을 루프형으로 바꾸어 준다:

    using Devectorize

    @devec r = exp(-abs(x-y))

포함조건 구문(comprehension syntax) 또한 디벡터라이제이션 계산을 위한 간결한 구문을 만들어준다:

    r = [exp(-abs(x[i]-y[i])) for i = 1:length(x)]

포함조건은 항상 결과를 저장하기 위해 새로운 배열을 만드는 것에 주의하자. 따라서, 사전에 할당된 배열에 결과를 작성하기 위해, 여전히 수동으로 계산을 디벡터라이제이션하거나  ``@devec`` 매크로를 사용한다.

## 계산을 단일 루프로 합한다 (Merge computations into a single loop)

배열을 훝어가는일, 특히 큰 배열의 경우, 캐시 누락이나 페이지 오류를 초래할 수 있고, 이는 성능을 꽤 지연시킨다. 따라서, 최대한 메모리 왕복 횟수를 최소화하는게 바람직하다.
예를 들면, 하나의 루프로 여러 맵핑을 계산할 수 있다:

    for i = 1:length(x)
        a[i] = x[i] + y[i]
        b[i] = x[i] - y[i]
    end

위와 같이 쓰는게 `a = x + y; b = x - y`와 같이 쓰는 것보다 빠르다.

다음 예제는 효율적으로 데이터 집합에 여러 통계치(예 : 합계, 최대, 최소,)를 계산하는 방법을 보여줍니다.

    n = length(x)
    rsum = rmax = rmin = x[1]
    for i = 2:n
        xi = x[i]
        rsum += xi
        if xi > rmax
            rmax = xi
        elseif xi < rmin
            rmin = xi
        end
    end

## 캐시 친화적인 코드를 쓰자 (Write cache-friendly codes)

현대 컴퓨터 시스템은 레지스터, 캐시의 여러 수준 및 RAM을 결합하여 복잡한 이기종 메모리 구조가있다. 캐시 계층 구조를 통해 데이터를 접근한다 -캐시는 자주 사용되는 데이터의 복사본을 저장하는 작고 더 빠른 메모리다.

대부분의 시스템은 직접 캐시 시스템을 제어하는 방법을 제공하지 않습니다. 그러나, 여러분이 **캐시 친화적인** 코드를 작성한다면 자동 캐시 관리 시스템이 일하기 쉽도록 도울 수 있다. 캐시 시스템의 작동 방식에 대한 모든 세부 사항을 이해 할 필요는 없다. 아래의 간단한 규칙을 준수하는 것이 충분하다:

> 데이터가 메모리에 상주하는 방법과 유사한 패턴으로 테이터에 접근하라- 메모리의 불연속적인 위치 사이를 이동하지 않는다.

이는 **국지성의 원칙**이다. 예를 들어, `x`가 연속 배열이라면, `x[i]`를 읽은 후에,  `x[i+1]`가 이미 캐시에 있을 가능성이 `x[i+1000000]`가 캐시될 경우보다 높다. 이 때는 당연히 `x[i+1]`에 접근하는게 `x[i+1000000]`보다 **훨씬** 빠르다.

![Column-major order: from Intel](http://software.intel.com/sites/products/documentation/hpc/mkl/mkl_userguide_lnx/GUID-3C2FFEF0-967E-43C3-992F-1ABEEE7C502E-low.jpg)

줄리아 배열은 열우선(column-major order)으로 저장되는데, 이는 한 열의 행들이 연속이라는 말이고, 한 행의 열은 그렇지 않다는 의미다. 따라서 행별 액세스보다 열(컬럼)별 액세스가  일반적으로 더 효율적이다.
행렬의 각 행의 합을 계산하는 문제를 생각해 볼 수 있다. 다음과 같이 구현을 보자:

    m, n = size(a)
    r = Array(Float64, m)

    for i = 1:m
        s = 0.
        for j = 1:n
            s += a[i,j]
        end
        r[i] = s
    end

여기에서 루프가 각 요소를 `a[i,1], a[i,2], ..., a[i,n]` 같이 행별로 액세스한다. 각 요소간의 간격은 `m`이다. 직관적으로, `m`길이의 보폭으로 각 행의 처음부터 끝까지 내부 루프에서 점프하고, 그뒤 다음 행의 처음으로 점프한다. 이는 비효율 적이고, 특히 `m`이 큰 숫자일 경우 더욱 그렇다.

이경우 계산 순서를 변경하면, 훨씬 더 캐시 친화적인 코드로 만들수 있다:

    for i = 1:m
        r[i] = a[i,1]
    end

    for j = 2:n, i = 1:m
        r[i] += a[i,j]
    end

일부 벤치마킹은 개선된 버전이 처음 버전보다 **5~10배**빨랐다.

## 루프내에서 배열을 만들지 마라(Avoid creating arrays in loops)

배열을 만드는 것은 메모리 할당을 요구하고 가비지 컬렉터의 부하를 준다. 같은 배열을 재사용하면 메모리 관리 비용을 줄인다.

축차 알고리즘을 사용할 때 배열을 업데이트하는 일은 드물지 않다. 예를 들어 K-mean알고리즘에서, 클러스터 평균들과 거리 둘다를 업데이트하길 때가 있다. 간단한 방법은 다음과 같다:

    while !converged && t < maxiter
        means = compute_means(x, labels)
        dists = compute_distances(x, means)
        labels = assign_labels(dists)
        ...
    end

위 구현에서 K-means은  `means`,`dists`,`labels`배열들이 각 축차에서 재생성된다. 각 단계에서의 이런 메모리 재할당은 불필요하다. 이들 배열의 크기는 고정되어, 그 저장소는 반복을 통해 재사용 할 수 있다. 다음은 대체 코드는 동일한 알고리즘을 구현하는 보다 효율적인 방법이다:

    d, n = size(x)

    # pre-allocate storage
    means = Array(Float64, d, K)
    dists = Array(Float64, K, n)
    labels = Array(Int, n)

    while !converged && t < maxiter
        update_means!(means, x, labels)
        update_distances!(dists, x, means)
        update_labels!(labels, dists)
        ...
    end

이 버전에서는, 함수가 미리 할당된 배열을 루프 내 업데이트에서 사용한다.

패키지를 작성하는 경우, 각 함수의 배열 출력 방식에 두가지 버전을 제공하는게 좋다: 하나는 미리 배열을 그자리에서 업데이트하는 것과 다른 것은 새로운 배열을 반환하는 것이다. 전자는 후자를 간단한 랩퍼로 싸서 입력 배열을 수정하기전에 복사하는 방식으로 만들수 있다.
좋은 예제는 [*Distributions.jl*](https://github.com/JuliaStats/Distributions.jl) 패키지로, `logpdf` 와 `logpdf!` 둘다 제공해서, 새 배열로 받고 싶을 때는 `lp = logpdf(d,x)`로,  `logpdf!(lp,d,x)`는 `lp`가 미리 할당되었을 때 사용한다.

## BLAS를 사용할 기회를 잡아라 (Identify opportunities to use BLAS)

줄리아는 선형대수 계산을 위해 다수의 [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) 루틴을 래핑하고 있다. 이들 루틴들은 빠른 수치 계산에 관한 세계최고의 전문가들에 의해 수십년간 연구되고 최적화된 결과물이다. 그결과로 가능한 부분에 루팅을 사용하면 마법과도 같은 성능 향상을 보인다. – BLAS 루튼들은 종종 간단한 루프 구현보다 몇 차수 빠르다.

예를 들어, 다음과 같은 벡터가중 구현을 보자:

    r = zeros(size(x,1))
    for j = 1:size(x,2)
        r += x[:,j] * w[j]
    end

 `r += x[:,j] * w[j]` 부분을 BLAS 명령어 `axpy!` 함수로 바꾸면 나은 성능을 보인다:

    for j = 1:size(x,2)
        axpy!(w[j], x[:,j], r)
    end

이서도 여전히 최적은 아니다. 선형대수에 익숙하다면, 행렬-벡터곱셈을 찾을수 있고, `r = x * w` 로 쓸수 있으며, 짧고 간단하고 명확할뿐만 아니라 이전의 다른 두 버전 보다 훨씬 빠르게 실행된다.

다음 예제는 두 행렬에서 열사이의 쌍 유클리드 거리를 계산하는 BLAS 루틴의 미묘한 응용 프로그램이다. 직접적으로 쌍 거리를 계산하는 간단한 구현은 다음과 같다:

    m, n = size(a)
    r = Array(Float64, m, n)

    for j = 1:n, i = 1:m
        r[i,j] = sqrt(sum(abs2(a[:,i] - b[:,j])))
    end

분명 최적은 아니다. 많은 입시 배열이 내부 루프의 계산식 평가도중 만들어진다. 속도를 높이기 위해 내부 표현식을 디벡터라이제이션 해보자:

    d, m = size(a)
    n = size(b,2)
    r = Array(Float64, m, n)

    for j = 1:n, i = 1:m
            s = 0.
            for k = 1:d
                s += abs2(a[k,i] - b[k,j])
            end
            r[i,j] = sqrt(s)
        end
    end

이 버전은 벡터화된 형태보다 훨씬 효율이 좋다. 그러나 좀더 나아가보자. 다른 전략을 선택하면, 거리 쌍을 계산하는 좀 더 빠른 알고리즘을 작성할수 있다. 그 트릭은 바로 두 벡터사이의 유클리드 거리의 제곱을 다음과 같이 풀어 쓰는 것이다:

    sum(abs2(x-y)) == sum(abs2(x)) + sum(abs2(y)) - 2*dot(x,y)

개별적으로 우변의 세가지 항을 계산한다면, BLAS 루틴에 완벽하게 호환된다. 아래, 놈(norm)을 래핑하는  [*NumericExtensions.jl*](https://github.com/lindahua/NumericExtensions.jl) 패키지를 사용하여 BLAS 루틴만으로 거리 쌍계산을 하는 구현을 보여준다 :

    using NumericExtensions   # for sqsum
    using Base.LinAlg.BLAS    # for gemm!

    m, n = size(a)

    sa = sqsum(a, 1)   # sum(abs2(x)) for each column in a
    sb = sqsum(b, 1)   # sum(abs2(y)) for each column in b

    r = sa .+ reshape(sb, 1, n)          # first two terms
    gemm!('T', 'N', -2.0, a, b, 1.0, r)  # add (-2.0) * a' * b to r

    for i = 1:length(r)
        r[i] = sqrt(r[i])
    end

개선 버전은  **100 배** 빠르다— BLAS `gemm` 함수는 지난 수십년간 여러 재능있는 엔지니어와 개발자들이 최적화 해왔다. 

여러분이 직접 이를 구현할 필요는 없다: [*Distance.jl*](https://github.com/lindahua/Distance.jl) 패키지가 이미 다양한, 이를 포함한 거리 메트릭 구현하고 있다. 여기서 보여주고자 한것은 가능한 모든 곳에서 BLAS 루틴을 사용함으로서 얻을수 있는 엄청난 성능 향상이다.

## 여러 패키지를 시도라하 (Explore available packages)

줄리아는 활발한 오픈 소스 생태계를 가고 있다. 다양한 패키지가 고성능 계산에 최적화된 알고리즘을 제공하기 위해 계발 되어 왔다.
직접 만들기로 마음먹기 전에 먼저 여러분의 요구를 충족하는 패키지 없나 찾아보고, 없으면 만들어서 기여할수 있다.

고성능 계산에 관심있는 이들을 위한 패키지는 다음과 같다:

* [NumericExtensions.jl](https://github.com/lindahua/NumericExtensions.jl) – 고성능 계산을 위한 줄리아의 기본 기능 확장 (추후에 기본 줄리아로 편입) 

* [Devectorize.jl](https://github.com/lindahua/Devectorize.jl) – 디벡터라이제이션을 위한 함수와 매크로 패키지. 고수준의 벡터화된 표현식을 작성하면서, 디벡터라이제션을 맛볼수 있게 해준다.

[Julia package list](http://docs.julialang.org/en/latest/packages/packagelist/) 에서 더 많은 패키지를 확인해보자. 줄리아는 또한  [sampling profiler](http://docs.julialang.org/en/latest/stdlib/profile/) 프로파일러를 제공하니, 여러분의 코드가 어느 곳에서 병목이 있는지 어림 짐작이 아니라 측정할 수 있다!
