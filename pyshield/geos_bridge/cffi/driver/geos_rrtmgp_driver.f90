program geos_rrtmgp_driver

  ! Standalone Fortran driver for the radiation CFFI bridge (Phase 1), mirroring
  ! the gtFV3 bridge's driver/gtfv3_driver.f90. It fills synthetic whole-tile
  ! inputs, drives init -> run -> finalize through the compiled
  ! libgeos_rrtmgp_interface_py.so, and checks the returned fluxes are finite and
  ! physically plausible. It needs NO staged data (the profile is synthesized).
  !
  ! The synthetic profile matches tests/geos_bridge/test_geos_rrtmgp_bridge.py:
  ! top-down (index 1 = TOA) geometric pressure 50 Pa -> 1.0e5 Pa, T=250 K layers,
  ! Tsfc=288 K, q=1e-3, o3=1e-7, co2=400 ppm, clouds zero, ocean (islmsk=0).
  !
  ! Field flat layout (the Phase-1 contract, see radiation_f_py_conversion.py):
  ! C-contiguous (ncol, nlev) => element (c, l) at index (c-1)*nlev + l.

  use iso_c_binding, only: c_int, c_double
  use ieee_arithmetic, only: ieee_is_finite
  use ieee_exceptions, only: ieee_get_halting_mode, ieee_set_halting_mode, ieee_all
  use geos_rrtmgp_interface_mod, only: &
       geos_rrtmgp_interface_f_init, &
       geos_rrtmgp_interface_f, &
       geos_rrtmgp_interface_f_finalize

  implicit none

  include 'mpif.h'

  integer, parameter :: NX = 4, NY = 4, NZ = 24, NHALO = 3
  integer, parameter :: NCOL = NX * NY, NLEV = NZ + 1
  integer, parameter :: TOP_AT_1 = 1
  real(c_double), parameter :: DT = 3600.0_c_double

  integer :: irank, nranks, mpierr
  integer :: c, l, idx
  logical :: halting_mode(5)
  real(c_double) :: p_lev(NLEV), p_lay(NZ)

  ! Inputs
  real(c_double) :: prsi(NCOL*NLEV), prsl(NCOL*NZ), tlyr(NCOL*NZ), tsfc(NCOL)
  real(c_double) :: qvapor(NCOL*NZ), qo3mr(NCOL*NZ), co2(NCOL*NZ)
  real(c_double) :: qliquid(NCOL*NZ), qice(NCOL*NZ), qcld(NCOL*NZ)
  real(c_double) :: sfc_tsfc(NCOL), islmsk(NCOL)

  ! Outputs
  real(c_double) :: flwu(NCOL*NLEV), flwd(NCOL*NLEV), fswu(NCOL*NLEV), fswd(NCOL*NLEV)
  real(c_double) :: flwu_clr(NCOL*NLEV), flwd_clr(NCOL*NLEV)
  real(c_double) :: fswu_clr(NCOL*NLEV), fswd_clr(NCOL*NLEV)
  real(c_double) :: fswn(NCOL)
  real(c_double) :: hrtlw(NCOL*NZ), hrtsw(NCOL*NZ), hrtlw_clr(NCOL*NZ), hrtsw_clr(NCOL*NZ)

  call MPI_Init(mpierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, mpierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, irank, mpierr)

  ! --- Synthetic top-down column profile (same on every column) ---
  do l = 1, NLEV
     p_lev(l) = 50.0_c_double * (1.0e5_c_double / 50.0_c_double) &
          ** (real(l - 1, c_double) / real(NZ, c_double))
  end do
  do l = 1, NZ
     p_lay(l) = 0.5_c_double * (p_lev(l) + p_lev(l + 1))
  end do

  do c = 1, NCOL
     do l = 1, NLEV
        idx = (c - 1) * NLEV + l
        prsi(idx) = p_lev(l)
     end do
     do l = 1, NZ
        idx = (c - 1) * NZ + l
        prsl(idx)    = p_lay(l)
        tlyr(idx)    = 250.0_c_double
        qvapor(idx)  = 1.0e-3_c_double
        qo3mr(idx)   = 1.0e-7_c_double
        co2(idx)     = 400.0e-6_c_double
        qliquid(idx) = 0.0_c_double
        qice(idx)    = 0.0_c_double
        qcld(idx)    = 0.0_c_double
     end do
     tsfc(c)     = 288.0_c_double
     sfc_tsfc(c) = 288.0_c_double
     islmsk(c)   = 0.0_c_double   ! ocean
  end do

  ! Zero the outputs so a short copy-back would be visibly detectable.
  flwu = -1.0_c_double; flwd = -1.0_c_double; fswu = -1.0_c_double; fswd = -1.0_c_double
  flwu_clr = -1.0_c_double; flwd_clr = -1.0_c_double
  fswu_clr = -1.0_c_double; fswd_clr = -1.0_c_double
  fswn = -1.0_c_double
  hrtlw = -1.0_c_double; hrtsw = -1.0_c_double
  hrtlw_clr = -1.0_c_double; hrtsw_clr = -1.0_c_double

  ! --- init: FPE-trap workaround (mirror FV_StateMod.F90:1206-1212). Importing
  !     numpy / pyRTE can raise SIGFPE under GEOS trapping; disable trapping
  !     across the Python init, then restore. ---
  call ieee_get_halting_mode(ieee_all, halting_mode)
  call ieee_set_halting_mode(ieee_all, .false.)
  call geos_rrtmgp_interface_f_init( &
       MPI_COMM_WORLD, &
       NX, NY, NZ, NHALO, TOP_AT_1, &
       2020, 1, 1, 12, 0, 0, DT)
  call ieee_set_halting_mode(ieee_all, halting_mode)

  ! --- run ---
  call geos_rrtmgp_interface_f( &
       MPI_COMM_WORLD, &
       NX, NY, NZ, TOP_AT_1, &
       2020, 1, 1, 12, 0, 0, &
       prsi, prsl, tlyr, tsfc, &
       qvapor, qo3mr, co2, &
       qliquid, qice, qcld, &
       sfc_tsfc, islmsk, &
       flwu, flwd, fswu, fswd, &
       flwu_clr, flwd_clr, fswu_clr, fswd_clr, &
       fswn, &
       hrtlw, hrtsw, hrtlw_clr, hrtsw_clr)

  ! --- finalize ---
  call geos_rrtmgp_interface_f_finalize()

  ! --- checks ---
  if (irank == 0) then
     print *, 'LW up   sfc / toa :', flwu((NLEV)), flwu(1)
     print *, 'LW down sfc / toa :', flwd((NLEV)), flwd(1)
     print *, 'SW down sfc / toa :', fswd((NLEV)), fswd(1)
     print *, 'sum(flwu), sum(flwd):', sum(flwu), sum(flwd)
     print *, 'sum(fswu), sum(fswd):', sum(fswu), sum(fswd)

     call assert_all_finite('flwu', flwu)
     call assert_all_finite('flwd', flwd)
     call assert_all_finite('fswu', fswu)
     call assert_all_finite('fswd', fswd)
     call assert_all_finite('hrtlw', hrtlw)
     call assert_all_finite('hrtsw', hrtsw)

     ! Physical sanity: surface downwelling LW must be positive at Tsfc=288 K.
     if (flwd(NLEV) <= 0.0_c_double) then
        print *, 'FAIL: surface LW down is not positive:', flwd(NLEV)
        call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
     end if
     print *, 'PASS: fluxes finite and surface LW down positive.'
  end if

  call MPI_Finalize(mpierr)

contains

  subroutine assert_all_finite(name, arr)
    character(len=*), intent(in) :: name
    real(c_double), intent(in) :: arr(:)
    integer :: i
    do i = 1, size(arr)
       if (.not. ieee_is_finite(arr(i))) then
          print *, 'FAIL: non-finite value in ', name, ' at ', i
          call MPI_Abort(MPI_COMM_WORLD, 2, i)
       end if
    end do
  end subroutine assert_all_finite

end program geos_rrtmgp_driver
