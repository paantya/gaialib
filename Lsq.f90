module LSQ

IMPLICIT NONE

contains

	Subroutine LSQM(a,y,w, x,d, s, r)

! m - количество уравнений
! n - количество неизвестных
! a(m,n) - матрица плана
! y(m) - столбец правых частей,  w(m) -  столбец весов;
! x(n) - ответ, d(n) - среднеквадратичные ошибки x;
!  s - среднеквадратичная ошибка единицы веса;
!  r(n,n) - корреляционная матрица.

	real(8), intent(in)  :: a(:,:), y(:), w(:)
	real(8), intent(out) :: x(:), d(:), s, r(:,:)

	integer i,j,k
    real(8) :: u
    real(8) :: c(size(x))

	integer :: m,n

	m=size(a, dim=1)
	n=size(a, dim=2)

	if (size(y)/=m .or. size(w)/=m .or. size(x)/=n .or. size(d)/=n .or. size(r,dim=1)/=n .or. size(r,dim=2) /= n) then
	 STOP 'Incorrect parameters of LSQM'
	endif

      do i=1,n

       do j=1,i
        u=0.0_8
        do k=1,m
	     u=u+a(k,i)*a(k,j)*w(k)
	    end do
        r(i,j)=u; r(j,i)=u
       end do

	   u=0.0_8
       do k=1,m
        u=u+a(k,i)*y(k)*w(k)
       end do
       c(i)=u

	  end do ! i=1,n

      call Invert(r)
      call Multiply(r,c,x)

      s=0.0_8
      do k=1,m
       u=0.0_8
	   do i=1,n
        u=u+a(k,i)*x(i)
	   enddo
       s=s+(u-y(k))**2 * w(k)
	  enddo
      s=sqrt(s/(m-n))

	  do i=1,n
       d(i)=s*sqrt(r(i,i))
	  enddo

      do i=1,n
       do j=1,i-1
          r(i,j)=r(i,j)/sqrt(r(i,i)*r(j,j))
          r(j,i)=r(i,j)
       enddo
	  enddo
      do i=1,n
       r(i,i)=1.0_8
	  enddo
	
	end subroutine

     Subroutine Multiply(a,b,c)
! Умножение матрицы на вектор c[m]:=a[m,n]*b[n]
	real(8), intent(in)  :: a(:,:),b(:)
	real(8), intent(out) :: c(:)
	integer m,n
	integer i,k
    real(8) :: t

    m=size(a,dim=1);	n=size(a,dim=2)
	if (size(b)/=n .or. size(c)/=m) then
	 STOP 'Incompatible size in Multiply'
	endif
      
      do i=1,m
       t=0.0
       do k=1,n
        t=t+a(i,k)*b(k)
       enddo
       c(i)=t
	enddo

	end subroutine


	Subroutine Invert(D) ! Обращение матрицы
	real(8), intent(inout) :: D(:,:)			 ! Исходная матрица	
	integer :: N2,i,j,k,is
	real(8) :: t
	real(8) a( size(D,dim=1), 2*size(D,dim=1) )
	integer :: N
	N=size(D,dim=1)
	N2=2*N

!		Наращивание с единичной матрицей
	
	DO i=1,N
	 DO j=1,N
	  a(i,j)=D(i,j)
	 END DO
	END DO
	DO i=1,N
	 DO j=1,N
	  IF (i==j) THEN
	            a(i,j+N)=1
			   ELSE
				a(i,j+N)=0
	  END IF
	 END DO
	END DO
!	Начало обращения
	DO i=1,N
	 k=i
	 DO WHILE (a(k,i)==0)
	  is=1
	  IF (k<N) THEN
	              K=K+1
			   ELSE	
				STOP 'Matrix is particular'	
	  ENDIF
	 END DO ! WHILE

	 IF (is==1) THEN
	  t=a(i,j)
	  a(k,j)=a(i,j)
	  a(i,j)=t
	 ENDIF

	 DO j=N2,i,-1
	  a(i,j)=a(i,j)/a(i,i)
	 ENDDO

	 DO k=1,N
	  IF (k /= i) THEN
	   DO j=N2,i,-1
	     a(k,j)=a(k,j)-a(i,j)*a(k,i)
	   ENDDO
	  ENDIF
	 ENDDO

	ENDDO
	
	DO i=1,N
	 DO j=1,N
	  D(i,j)=a(i,j+N)
	 ENDDO
	ENDDO

  	END Subroutine
  	
  	
 REAL(4) Function Ranorm(s)
 REAL(4), INTENT(IN) :: S
 REAL(4) :: a,x 
 INTEGER(4) :: i 
  a=0
  DO I=1, 12 
   !CALL RANDOM(x)
   CALL RANDOM_NUMBER(X)
   a=a+x
  END DO 
  Ranorm=(a-6.0)*s
 END FUNCTION RANORM

 	  
SUBROUTINE INVMATRIX(A)

      IMPLICIT NONE
      REAL*8    :: A(:,:)
      INTEGER   :: MMM
      REAL*8                :: C0,C1,SIGMA
      INTEGER               :: I,J,K,M,METHOD,N1,N2
      INTEGER               :: LLL

      Real*8, Allocatable   :: WK(:,:), BK(:,:), E(:,:), U(:,:)
      Real*8, Allocatable   :: VM(:),WM(:)

      MMM=size(A,Dim=1)

!  MMM=size(WK,Dim=1)         ! MMM is the number of rows
!  LLL=size(WK,Dim=2)         ! LLL is the number of columns


      N1=MMM
      LLL=2*MMM
      N2=LLL	    
      Allocate( WK(N1,N2), BK(N1,N2),E(N1,N1),U(N1,N1) ) 
      Allocate( VM(N1), WM(N1) )

      DO I=1,MMM
        DO J=1,MMM
          E(I,J)=0.0D+00
        ENDDO
        E(I,I)=1.0D+00
      ENDDO
 
      DO I=1,MMM
        DO J=1,MMM
          WK(I,J)=A(I,J)
          WK(I,J+MMM)=E(I,J)
        ENDDO
      ENDDO
 

      DO 1000 M=1,MMM-1

        METHOD=1
        DO I=M+1,MMM
          IF (WK(I,M).NE.0.0D+00) METHOD=0
        ENDDO
        IF (METHOD.EQ.1) THEN
          C0=WK(M,M)
          DO I=M,LLL
            WK(M,I)=WK(M,I)/C0
          ENDDO
          GOTO 1000
        ELSE
          DO I=1,MMM
            VM(I)=0.0D+00
          ENDDO
          DO I=M,MMM
            VM(I)=WK(I,M)
          ENDDO
          C0=0.0D+00
          DO I=M,MMM
            C0=C0+VM(I)*VM(I)
          ENDDO
          IF (VM(M).GE.0.0D+00) THEN
            SIGMA=DSQRT(C0)
          ELSE
            SIGMA=-DSQRT(C0)
          END IF
          DO I=1,MMM
            WM(I)=VM(I)
          ENDDO
          WM(M)=WM(M)+SIGMA
          C0=0.0D+00
          DO I=M,MMM
            C0=C0+WM(I)*WM(I)
          ENDDO
          C1=DSQRT(C0)
          DO I=M,MMM
            WM(I)=WM(I)/C1
          ENDDO
          DO I=1,MMM
            DO J=1,MMM
              U(I,J)=E(I,J)-2.0D+00*WM(I)*WM(J)
            ENDDO
          ENDDO
          DO I=1,MMM
            DO J=1,LLL
              C0=0.0D+00
              DO K=1,MMM
                C0=C0+U(I,K)*WK(K,J)
              ENDDO
              BK(I,J)=C0
            ENDDO
          ENDDO
          DO I=1,MMM
            DO J=1,LLL
              WK(I,J)=BK(I,J)
            ENDDO
          ENDDO
          C0=WK(M,M)
          DO I=M,LLL
            WK(M,I)=WK(M,I)/C0
          ENDDO
        END IF
1000  CONTINUE

      C0=WK(MMM,MMM)
      DO I=MMM,LLL
        WK(MMM,I)=WK(MMM,I)/C0
      ENDDO


      DO K=MMM,2,-1
        DO I=K-1,1,-1
          C0=WK(I,K)
          DO J=K,LLL
            WK(I,J)=WK(I,J)-WK(K,J)*C0
          ENDDO
        ENDDO
      ENDDO  		    


      DO I=1,MMM
        DO J=1,MMM
          A(I,J)=WK(I,J+MMM)
        ENDDO
      ENDDO
 

      DEAllocate( WK, BK,E,U ) 
      DEAllocate( VM, WM )
 

END SUBROUTINE INVMATRIX


REAL(8) FUNCTION DETERMINANT(A)
REAL(8),INTENT(IN)    :: A(:,:)
INTEGER(4)   :: N
REAL(8)    :: DET
REAL*8               :: T,D,MAX
REAL*8               :: B(size(A,Dim=1),size(A,Dim=1))
INTEGER              :: I,J,K

N=size(A,Dim=1)

DO I=1,N
  DO J=1,N
    B(I,J)=A(I,J)
  ENDDO
ENDDO

D=1.0D+00

DO K=1,N
  MAX=0.0D+00
  DO I=K,N
    T=B(I,K)
    IF (DABS(T).GT.DABS(MAX)) THEN
      MAX=T
      J=I
    END IF
  END DO
  IF (MAX.EQ.0.0D+00) THEN
    D=0.0D+00
    GOTO 100
  END IF
  IF (J.NE.K) THEN
    D=-D
    DO I=K,N
      T=B(J,I)
      B(J,I)=B(K,I)
      B(K,I)=T
    END DO
  END IF
  DO I=K+1,N
    T=B(I,K)/MAX
    DO J=K+1,N
      B(I,J)=B(I,J)-T*B(K,J)
    END DO
  END DO
  D=D*B(K,K)
END DO

100 DET=D

DETERMINANT=DET

END FUNCTION DETERMINANT

 	

end module LSQ