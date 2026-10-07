module geos_rrtmgp_interface_mod

  ! Fortran bind(c) interface for the GEOS <-> PySHiELD radiation CFFI bridge
  ! (Phase 1). Mirrors the merged gtFV3 bridge's geos_gtfv3_interface.f90:
  ! scalars are passed `value, intent(in)`; whole-tile field arrays are passed
  ! as `dimension(*)`. The one deliberate change from gtFV3 is dtype: RRTMGP is
  ! double precision (R8), so every field array is real(c_double), NOT c_float.
  !
  ! The C symbols bound here (geos_rrtmgp_interface_c*) are implemented in
  ! geos_rrtmgp_interface.c, which MPI_Comm_f2c's the comm and calls the
  ! CFFI-exported geos_rrtmgp_interface_py* trampolines.

  use iso_c_binding, only: c_int, c_double

  implicit none

  private
  public :: geos_rrtmgp_interface_f_init
  public :: geos_rrtmgp_interface_f
  public :: geos_rrtmgp_interface_f_finalize

  interface

     subroutine geos_rrtmgp_interface_f_init( &
          comm, &
          nx, ny, nz, nhalo, top_at_1, &
          yy, mm, dd, hh, mn, sc, dt &
          ) bind(c, name='geos_rrtmgp_interface_c_init')

       import c_int, c_double

       implicit none
       integer(kind=c_int), value, intent(in) :: comm
       integer(kind=c_int), value, intent(in) :: nx, ny, nz, nhalo
       integer(kind=c_int), value, intent(in) :: top_at_1       ! 1 = top-down (GEOS)
       integer(kind=c_int), value, intent(in) :: yy, mm, dd, hh, mn, sc
       real(kind=c_double), value, intent(in) :: dt             ! radiation timestep [s]

     end subroutine geos_rrtmgp_interface_f_init

     subroutine geos_rrtmgp_interface_f( &
          ! Input scalars
          comm, &
          nx, ny, nz, top_at_1, &
          yy, mm, dd, hh, mn, sc, &

          ! Input: RTE_RRTMGPState base fields (whole-tile, R8)
          prsi, prsl, tlyr, tsfc, &
          qvapor, qo3mr, co2, &
          qliquid, qice, qcld, &

          ! Input: SurfaceState fields
          sfc_tsfc, islmsk, &

          ! Output: fluxes (level) + net-sfc SW (col) + heating rates (layer)
          flwu, flwd, fswu, fswd, &
          flwu_clr, flwd_clr, fswu_clr, fswd_clr, &
          fswn, &
          hrtlw, hrtsw, hrtlw_clr, hrtsw_clr &
          ) bind(c, name='geos_rrtmgp_interface_c')

       import c_int, c_double

       implicit none

       ! Input scalars
       integer(kind=c_int), value, intent(in) :: comm
       integer(kind=c_int), value, intent(in) :: nx, ny, nz, top_at_1
       integer(kind=c_int), value, intent(in) :: yy, mm, dd, hh, mn, sc

       ! Input field arrays (sized by caller: level=nz+1, layer=nz, col=ncol)
       real(kind=c_double), dimension(*), intent(in) :: prsi, prsl, tlyr, tsfc
       real(kind=c_double), dimension(*), intent(in) :: qvapor, qo3mr, co2
       real(kind=c_double), dimension(*), intent(in) :: qliquid, qice, qcld
       real(kind=c_double), dimension(*), intent(in) :: sfc_tsfc, islmsk

       ! Output field arrays (driver writes; marshalled back out)
       real(kind=c_double), dimension(*), intent(inout) :: flwu, flwd, fswu, fswd
       real(kind=c_double), dimension(*), intent(inout) :: flwu_clr, flwd_clr, fswu_clr, fswd_clr
       real(kind=c_double), dimension(*), intent(inout) :: fswn
       real(kind=c_double), dimension(*), intent(inout) :: hrtlw, hrtsw, hrtlw_clr, hrtsw_clr

     end subroutine geos_rrtmgp_interface_f

     subroutine geos_rrtmgp_interface_f_finalize() &
          bind(c, name='geos_rrtmgp_interface_c_finalize')
     end subroutine geos_rrtmgp_interface_f_finalize

  end interface

end module geos_rrtmgp_interface_mod
