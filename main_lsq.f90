use LSQ

integer n,m
parameter (n = 1200, m = 49)

real(8) :: a(0:n-1,0:m-1), y(0:n-1), w(0:n-1)
real(8) :: x(0:m-1), d(0:m-1), s, r(0:m-1,0:m-1)

w = 1

open(10,file = "a.in")

read(10,*) a
close(10)
!print*, a

open(20,file = "y.in")

read(20,*) y
!print*, y
close(20)

call LSQM(a,y,w, x,d, s, r)

open(30,file = "x.out")
write(30,*), x
close(30)


open(40,file = "d.out")
write(40,*), d
close(40)
end