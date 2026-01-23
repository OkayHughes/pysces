# ! ----------------------------------------------------------------------
# ! Public API.

# subroutine gfr_init(par, elem, nphys, check, boost_pg1)
#   ! Initialize the gfr internal data structure.
#   !   nphys is N in pgN.
#   !   check is optional and defaults to 0, no checking. It will produce very
#   ! verbose output if something goes wrong. It is also expensive. It is
#   ! intended to be used in unit testing and (if ever needed) for
#   ! debugging. There are three levels: 0, no checking; 1, global properties
#   ! only; 2, also element-local properties.

#   use kinds, only: iulog
#   use dimensions_mod, only: nlev
#   use parallel_mod, only: parallel_t, abortmp
#   use quadrature_mod, only: gausslobatto, quadrature_t
#   use control_mod, only: geometry, cubed_sphere_map

#   type (parallel_t), intent(in) :: par
#   type (element_t), intent(in) :: elem(:)
#   integer, intent(in) :: nphys
#   integer, intent(in), optional :: check
#   logical, intent(in), optional :: boost_pg1

#   real(real_kind) :: R(npsq,nphys_max*nphys_max), tau(npsq)
#   integer :: nphys2

#   gfr%check = 0
#   if (present(check)) gfr%check = check
#   gfr%check_ok = .true.

#   gfr%boost_pg1 = .false.
#   if (nphys == 1 .and. present(boost_pg1)) gfr%boost_pg1 = boost_pg1    

#   gfr%tolfac = one
#   if (par%masterproc) then
#      write(iulog,*) 'gfr> Running with dynamics and physics on separate grids (physgrid).'
#      write(iulog, '(a,i3,a,i2,a,l2)') 'gfr> init nphys', nphys, ' check', gfr%check, &
#           ' boost_pg1', gfr%boost_pg1
#      if (nphys == 1) then
#         ! Document state of pg1. dcmip2016_test1 shows it is too coarse. For
#         ! boost_pg1 = true, stepon's DSS loop needs to be separated from its
#         ! tendency application loop.
#         write(iulog,*) 'gfr> Warning: pg1 is too coarse; see comments at top of gllfvremap_mod.F90'
#         if (.not. gfr%boost_pg1) then
#            write(iulog,*) 'gfr> Warning: If you want to try pg1, use the boosted-accuracy &
#                 &boost_pg1 option and call gfr_pg1_reconstruct(_topo).'
#         end if
#      end if
#   end if

#   if (nphys > np) then
#      ! The FV -> GLL map is defined only if nphys <= np. If we ever are
#      ! interested in the case of nphys > np, we will need to write a different
#      ! map. See "!assume" annotations for mathematical assumptions in
#      ! particular routines.
#      call abortmp('gllfvremap_mod: nphys must be <= np')
#   end if
#   if (qsize == 0) then
#      call abortmp('gllfvremap_mod: qsize must be >= 1')
#   end if

#   gfr%have_fv_topo_file_phis = .false.
#   gfr%nphys = nphys
#   ! npi is the internal GLL np parameter. The high-order remap operator remaps
#   ! from FV to npi-GLL grids, then interpolates from npi-GLL to np-GLL
#   ! grids. In the case of nphys=1, npi must be 2 for GLL to make sense.
#   gfr%npi = max(2, nphys)
#   nphys2 = nphys*nphys

#   gfr%is_planar = trim(geometry) == 'plane'

#   call gfr_init_w_gg(np, gfr%w_gg)
#   call gfr_init_w_gg(gfr%npi, gfr%w_sgsg)
#   call gfr_init_w_ff(nphys, gfr%w_ff)
#   call gfr_init_M_gf(np, nphys, gfr%M_gf, .true.)
#   gfr%g2f_remapd(:,:,:nphys2) = reshape(gfr%M_gf(:,:,:nphys,:nphys), (/np,np,nphys2/))
#   call gfr_init_M_gf(gfr%npi, nphys, gfr%M_sgf, .false.)
#   call gfr_init_R(gfr%npi, nphys, gfr%w_sgsg, gfr%M_sgf, R, tau)
#   call gfr_init_interp_matrix(gfr%npi, gfr%interp)
#   call gfr_init_f2g_remapd(gfr, R, tau)

#   allocate(gfr%fv_metdet(nphys2,nelemd), &
#        gfr%D_f(nphys2,2,2,nelemd), gfr%Dinv_f(nphys2,2,2,nelemd), &
#        gfr%qmin(nlev,max(1,qsize),nelemd), gfr%qmax(nlev,max(1,qsize),nelemd), &
#        gfr%phis(nphys2,nelemd), gfr%center_f(nphys,nphys,nelemd), &
#        gfr%corners_f(4,nphys,nphys,nelemd))
#   call gfr_init_geometry(elem, gfr)
#   call gfr_init_Dmap(elem, gfr)

#   if (nphys == 1 .and. gfr%boost_pg1) call gfr_pg1_init(gfr)

#   if (gfr%check > 0) call check_areas(par, gfr, elem, 1, nelemd)
# end subroutine gfr_init


# subroutine gfr_init_w_gg(np, w_gg)
#   ! Init GLL w(i)*w(j) values on the reference element.

#   use quadrature_mod, only : gausslobatto, quadrature_t
  
#   integer, intent(in) :: np
#   real(kind=real_kind), intent(out) :: w_gg(:,:)

#   type (quadrature_t) :: gll
#   integer :: i,j

#   gll = gausslobatto(np)

#   do j = 1,np
#      do i = 1,np
#         w_gg(i,j) = gll%weights(i)*gll%weights(j)
#      end do
#   end do
  
#   call gll_cleanup(gll)
# end subroutine gfr_init_w_gg


# subroutine gfr_init_w_ff(nphys, w_ff)
#   ! Init FV w(i)*w(j) values on the reference element.
  
#   integer, intent(in) :: nphys
#   real(kind=real_kind), intent(out) :: w_ff(:)

#   w_ff(:nphys*nphys) = four/real(nphys*nphys, real_kind)
# end subroutine gfr_init_w_ff



# subroutine gfr_init_M_gf(np, nphys, M_gf, scale)
#   ! Compute the mixed mass matrix with range the FV subcells and
#   ! domain the GLL nodes.

#   use quadrature_mod, only : gausslobatto, quadrature_t

#   integer, intent(in) :: np, nphys
#   real(kind=real_kind), intent(out) :: M_gf(:,:,:,:)
#   logical, intent(in) :: scale

#   type (quadrature_t) :: gll
#   integer :: gi, gj, fi, fj, qi, qj
#   real(kind=real_kind) :: xs, xe, ys, ye, ref, bi(np), bj(np)

#   gll = gausslobatto(np)

#   M_gf = zero

#   do fj = 1,nphys
#      ! The subcell is [xs,xe]x[ys,ye].
#      xs = two*real(fj-1, real_kind)/real(nphys, real_kind) - one
#      xe = two*real(fj, real_kind)/real(nphys, real_kind) - one
#      do fi = 1,nphys
#         ys = two*real(fi-1, real_kind)/real(nphys, real_kind) - one
#         ye = two*real(fi, real_kind)/real(nphys, real_kind) - one
#         ! Use GLL quadrature within this subcell.
#         do qj = 1,np
#            ! (xref,yref) are w.r.t. the [-1,1]^2 reference domain mapped to
#            ! the subcell.
#            ref = xs + half*(xe - xs)*(one + gll%points(qj))
#            call eval_lagrange_bases(gll, np, ref, bj)
#            do qi = 1,np
#               ref = ys + half*(ye - ys)*(one + gll%points(qi))
#               call eval_lagrange_bases(gll, np, ref, bi)
#               do gj = 1,np
#                  do gi = 1,np
#                     ! Accumulate each GLL basis's contribution to this
#                     ! subcell.
#                     M_gf(gi,gj,fi,fj) = M_gf(gi,gj,fi,fj) + &
#                          gll%weights(qi)*gll%weights(qj)*bi(gi)*bj(gj)
#                  end do
#               end do
#            end do
#         end do
#      end do
#   end do

#   M_gf = M_gf/real(nphys*nphys, real_kind)

#   if (scale) then
#      ! Scale so the sum over FV subcells gives the GLL weights to machine
#      ! precision.
#      do gj = 1,np
#         do gi = 1,np
#            M_gf(gi,gj,:nphys,:nphys) = M_gf(gi,gj,:nphys,:nphys)* &
#                 ((gll%weights(gi)*gll%weights(gj))/ &
#                 sum(M_gf(gi,gj,:nphys,:nphys)))
#         end do
#      end do
#   end if

#   call gll_cleanup(gll)
# end subroutine gfr_init_M_gf



# subroutine eval_lagrange_bases(gll, np, x, y)
#   ! Evaluate the GLL basis functions at x in [-1,1], writing the
#   ! values to y(1:np). This implements the Lagrange interpolant.

#   use quadrature_mod, only : quadrature_t
  
#   type (quadrature_t), intent(in) :: gll
#   integer, intent(in) :: np
#   real(kind=real_kind), intent(in) :: x ! in [-1,1]
#   real(kind=real_kind), intent(out) :: y(:)

#   integer :: i, j
#   real(kind=real_kind) :: f

#   do i = 1,np
#      f = one
#      do j = 1,np
#         if (j /= i) then
#            f = f*((x - gll%points(j))/(gll%points(i) - gll%points(j)))
#         end if
#      end do
#      y(i) = f
#   end do
# end subroutine eval_lagrange_bases


# subroutine gfr_init_R(np, nphys, w_gg, M_gf, R, tau)
#   ! We want to solve
#   !     min_g 1/2 g'M_gg g - g' M_gf f
#   !      st   M_gf' g = M_ff f,
#   ! which gives
#   !     [M_gg -M_gf] [g] = [M_gf f]
#   !     [M_gf'  0  ] [y]   [M_ff f].
#   ! Recall M_gg, M_ff are diag. Let
#   !     S = M_gf' inv(M_gg) M_gf.
#   ! Then
#   !     g = inv(M_gg) M_gf inv(S) M_ff f.
#   ! Compute the QR factorization sqrt(inv(M_gg)) M_gf = Q R so that S =
#   ! R'R. In this module, we can take M_gg = diag(w_gg) and M_ff = diag(w_ff)
#   ! with no loss of accuracy.
#   !   If nphys = np, then the problem reduces to
#   !     M_gf' g = M_ff f.
#   ! M_gf is symmetric. We could use the same computations as above
#   ! or, to gain a bit more accuracy, compute the simpler
#   !     M_gf = Q R
#   ! and later solve
#   !     R'Q' g = M_ff f.
#   !   In either case, all of this is one-time initialization; during
#   ! time stepping, just a matvec is computed.
#   !
#   !assume nphys <= np

#   integer, intent(in) :: np, nphys
#   real(kind=real_kind), intent(in) :: w_gg(:,:), M_gf(:,:,:,:)
#   real(kind=real_kind), intent(out) :: R(:,:), tau(:)

#   real(kind=real_kind) :: wrk(np*np*nphys*nphys), v
#   integer :: gi, gj, fi, fj, npsq, info

#   do fj = 1,nphys
#      do fi = 1,nphys
#         do gi = 1,np
#            do gj = 1,np
#               v = M_gf(gi,gj,fi,fj)
#               if (nphys < np) v = v/sqrt(w_gg(gi,gj))
#               R(np*(gi-1) + gj, nphys*(fi-1) + fj) = v
#            end do
#         end do
#      end do
#   end do
#   call dgeqrf(np*np, nphys*nphys, R, size(R,1), tau, wrk, np*np*nphys*nphys, info)
# end subroutine gfr_init_R



# subroutine gfr_init_interp_matrix(npsrc, interp)
#   ! Compute the matrix that interpolates from the npi-GLL nodes to
#   ! the np-GLL nodes.

#   use quadrature_mod, only : gausslobatto, quadrature_t

#   integer, intent(in) :: npsrc
#   real(kind=real_kind), intent(out) :: interp(:,:,:,:)

#   type (quadrature_t) :: glls, gllt
#   integer :: si, sj, ti, tj
#   real(kind=real_kind) :: bi(npsrc), bj(npsrc)

#   glls = gausslobatto(npsrc)
#   gllt = gausslobatto(np)

#   do tj = 1,np
#      call eval_lagrange_bases(glls, npsrc, real(gllt%points(tj), real_kind), bj)
#      do ti = 1,np
#         call eval_lagrange_bases(glls, npsrc, real(gllt%points(ti), real_kind), bi)
#         do sj = 1,npsrc
#            do si = 1,npsrc
#               interp(si,sj,ti,tj) = bi(si)*bj(sj)
#            end do
#         end do
#      end do
#   end do

#   call gll_cleanup(glls)
#   call gll_cleanup(gllt)
# end subroutine gfr_init_interp_matrix


#   subroutine gfr_init_f2g_remapd(gfr, R, tau)
#   ! Apply gfr_init_f2g_remapd_op to the Id matrix to get the remap operator's
#   ! matrix representation.

#   !assume nphys <= np

#   type (GllFvRemap_t), intent(inout) :: gfr
#   real(kind=real_kind), intent(in) :: R(:,:), tau(:)

#   integer :: nf, fi, fj, gi, gj
#   real(kind=real_kind) :: f(np,np), g(np,np)

#   gfr%f2g_remapd = zero
#   f = zero
#   nf = gfr%nphys
#   do fi = 1,nf
#      do fj = 1,nf
#         f(fi,fj) = one
#         call gfr_f2g_remapd_op(gfr, R, tau, f, g)
#         gfr%f2g_remapd(fi + (fj-1)*nf,:,:) = g
#         f(fi,fj) = zero
#      end do
#   end do
# end subroutine gfr_init_f2g_remapd

# subroutine gfr_f2g_remapd_op(gfr, R, tau, f, g)
#   ! This operator implements the linear operator that solves the
#   ! problem described in gfr_init_R.

#   !assume nphys <= np

#   type (GllFvRemap_t), intent(in) :: gfr
#   real(kind=real_kind), intent(in) :: R(:,:), tau(:), f(:,:)
#   real(kind=real_kind), intent(out) :: g(:,:)

#   integer :: nf, nf2, npi, np2, gi, gj, fi, fj, info
#   real(kind=real_kind) :: accum, wrk(gfr%nphys,gfr%nphys), x(np,np), wr(np*np)

#   nf = gfr%nphys
#   nf2 = nf*nf
#   npi = gfr%npi
#   np2 = np*np

#   ! Solve the constrained projection described in gfr_init_R:
#   !     g = inv(M_sgsg) M_sgf inv(S) M_ff f
#   wrk = reshape(gfr%w_ff(:nf2), (/nf,nf/))*f(:nf,:nf)
#   if (nf == npi) then
#      call dtrsm('L', 'U', 'T', 'N', nf2, 1, one, R, size(R,1), wrk, nf2)
#      call dormqr('L', 'N', nf2, 1, nf2, R, size(R,1), tau, wrk, nf2, wr, np2, info)
#      g(:npi,:npi) =  wrk
#   else
#      call dtrtrs('U', 'T', 'N', nf2, 1, R, size(R,1), wrk, nf2, info)
#      call dtrtrs('U', 'N', 'N', nf2, 1, R, size(R,1), wrk, nf2, info)
#      g(:npi,:npi) = zero
#      do fj = 1,nf
#         do fi = 1,nf
#            do gj = 1,npi
#               do gi = 1,npi
#                  g(gi,gj) = g(gi,gj) + gfr%M_sgf(gi,gj,fi,fj)*wrk(fi,fj)
#               end do
#            end do
#         end do
#      end do
#   end if
#   if (npi < np) then
#      if (nf == npi) then
#         x(:nf,:nf) = g(:nf,:nf)
#      else
#         ! Finish the projection:
#         !     wrk = inv(M_sgsg) g
#         do gj = 1,npi
#            do gi = 1,npi
#               x(gi,gj) = g(gi,gj)/gfr%w_sgsg(gi,gj)
#            end do
#         end do
#      end if
#      ! Interpolate from npi to np; if npi = np, this is just the Id matrix.
#      call apply_interp(gfr%interp, np, npi, x, g)
#   elseif (nf < npi) then
#      ! Finish the projection.
#      do gj = 1,np
#         do gi = 1,np
#            g(gi,gj) = g(gi,gj)/gfr%w_gg(gi,gj)
#         end do
#      end do
#   end if
# end subroutine gfr_f2g_remapd_op




#   subroutine gfr_init_geometry(elem, gfr)
#   use kinds, only: iulog
#   use control_mod, only: cubed_sphere_map
#   use coordinate_systems_mod, only: cartesian3D_t, spherical_polar_t, &
#        sphere_tri_area, change_coordinates
#   use cube_mod, only: ref2sphere
#   use physical_constants, only: dx, dy

#   type (element_t), intent(in) :: elem(:)
#   type (GllFvRemap_t), intent(inout) :: gfr

#   type (spherical_polar_t) :: p_sphere
#   type (cartesian3D_t) :: fv_corners_xyz(2,2), ctr
#   real(kind=real_kind) :: ones(np*np), ones2(np,np), ae(2), be(2), &
#        spherical_area, tmp, ac, bc
#   integer :: nf, nf2, ie, i, j, k, ai, bi, idx

#   nf = gfr%nphys
#   nf2 = nf*nf
#   do ie = 1,nelemd
#      do j = 1,nf
#         call gfr_f_ref_edges(nf, j, be)
#         call gfr_f_ref_center(nf, j, bc)
#         do i = 1,nf
#            call gfr_f_ref_edges(nf, i, ae)
#            call gfr_f_ref_center(nf, i, ac)
#            k = i+(j-1)*nf
#            ctr%x = zero; ctr%y = zero; ctr%z = zero
#            do bi = 1,2
#               do ai = 1,2
#                  if ( (i == 1  .and. ai == 1 .and. j == 1  .and. bi == 1) .or. &
#                       (i == 1  .and. ai == 1 .and. j == nf .and. bi == 2) .or. &
#                       (i == nf .and. ai == 2 .and. j == 1  .and. bi == 1) .or. &
#                       (i == nf .and. ai == 2 .and. j == nf .and. bi == 2)) then
#                     ! Use the element corner if we are at it.
#                     idx = 2*(bi-1)
#                     if (bi == 1) then
#                        idx = idx + ai
#                     else
#                        idx = idx + 3 - ai
#                     end if
#                     fv_corners_xyz(ai,bi) = elem(ie)%corners3D(idx)
#                  else
#                     if (gfr%is_planar) then
#                        call ref2plane(elem(ie)%corners3D, ae(ai), be(bi), &
#                             fv_corners_xyz(ai,bi))
#                     else
#                        ! fv_corners_xyz(ai,bi) contains the cartesian point
#                        ! before it's converted to lat-lon.
#                        p_sphere = ref2sphere(ae(ai), be(bi), elem(ie)%corners3D, &
#                             cubed_sphere_map, elem(ie)%corners, elem(ie)%facenum, &
#                             fv_corners_xyz(ai,bi))
#                        if (cubed_sphere_map == 0) then
#                           ! In this case, fv_corners_xyz above is not set.
#                           fv_corners_xyz(ai,bi) = change_coordinates(p_sphere)
#                        end if
#                     end if
#                  end if
#                  ctr%x = ctr%x + fv_corners_xyz(ai,bi)%x
#                  ctr%y = ctr%y + fv_corners_xyz(ai,bi)%y
#                  ctr%z = ctr%z + fv_corners_xyz(ai,bi)%z
#               end do
#            end do

#            if (gfr%is_planar) then
#               ! The cell area is (dx*dy)/nf^2. This must then be divided by
#               ! w_ff = 4/nf^2, leaving (dx*dy)/4.
#               gfr%fv_metdet(k,ie) = (dx*dy)/four
#            elseif (cubed_sphere_map == 2) then
#               call sphere_tri_area(fv_corners_xyz(1,1), fv_corners_xyz(2,1), &
#                    fv_corners_xyz(2,2), spherical_area)
#               call sphere_tri_area(fv_corners_xyz(1,1), fv_corners_xyz(2,2), &
#                    fv_corners_xyz(1,2), tmp)
#               spherical_area = spherical_area + tmp
#               gfr%fv_metdet(k,ie) = spherical_area/gfr%w_ff(k)
#            end if
#            if (gfr%is_planar .or. cubed_sphere_map == 2) then
#               ctr%x = ctr%x/four; ctr%y = ctr%y/four; ctr%z = ctr%z/four
#               if (cubed_sphere_map == 2) then
#                  ! [Projection bug 2023/11]: In the initial implementation, I
#                  ! forgot to project to the sphere. Mathematically, it doesn't
#                  ! matter: change_coordinates is invariant to the norm, and
#                  ! sphere2ref does the right thing since it solves a
#                  ! least-squares problem. In finite precision, fixing this bug
#                  ! would cause eps-level changes in ref2sphere and thus cause
#                  ! all pg2 tests to be non-BFB. I'm leaving it as is.
#               end if
#               gfr%center_f(i,j,ie) = ctr
#            end if

#            do bi = 1,2
#               do ai = 1,2
#                  ! CCW with ai the fast direction.
#                  idx = 2*(bi-1)
#                  if (bi == 1) then
#                     idx = idx + ai
#                  else
#                     idx = idx + 3 - ai
#                  end if
#                  gfr%corners_f(idx,i,j,ie) = fv_corners_xyz(ai,bi)
#               end do
#            end do
#         end do
#      end do
#   end do

#   if (cubed_sphere_map == 0 .and. .not. gfr%is_planar) then
#      ! For cubed_sphere_map == 0, we set the center so that it maps to the ref
#      ! element center and set fv_metdet so that it corresponds to the integral
#      ! of metdet over the FV subcell. TempestRemap establishes the
#      ! cubed_sphere_map == 2 convention, but for cubed_sphere_map == 0 there
#      ! is no external convention.
#      ones = one
#      ones2 = one
#      do ie = 1,nelemd
#         call gfr_g2f_remapd(gfr, elem(ie)%metdet, ones, ones2, gfr%fv_metdet(:nf2,ie))
#         do j = 1,nf
#            call gfr_f_ref_center(nf, j, bc)
#            do i = 1,nf
#               call gfr_f_ref_center(nf, i, ac)
#               p_sphere = ref2sphere(ac, bc, elem(ie)%corners3D, cubed_sphere_map, &
#                    elem(ie)%corners, elem(ie)%facenum)
#               gfr%center_f(i,j,ie) = change_coordinates(p_sphere)
#            end do
#         end do
#      end do
#   end if

#   ! Make the spherical area of the element according to FV and GLL agree to
#   ! machine precision.
#   if (gfr%check > 0) allocate(gfr%check_areas(1,nelemd))
#   do ie = 1,nelemd
#      if (gfr%check > 0) gfr%check_areas(1,ie) = sum(gfr%w_ff(:nf2)*gfr%fv_metdet(:nf2,ie))
#      gfr%fv_metdet(:nf2,ie) = gfr%fv_metdet(:nf2,ie)* &
#           (sum(elem(ie)%spheremp)/sum(gfr%w_ff(:nf2)*gfr%fv_metdet(:nf2,ie)))
#   end do
# end subroutine gfr_init_geometry



#   subroutine gfr_f_ref_center(nphys, i, a)
#   ! FV subcell center in ref [-1,1]^2 coord.

#   integer, intent(in) :: nphys, i
#   real(kind=real_kind), intent(out) :: a

#   a = two*((real(i-1, real_kind) + half)/real(nphys, real_kind)) - one
# end subroutine gfr_f_ref_center

# subroutine gfr_f_ref_edges(nphys, i_fv, a)
#   ! FV subcell edges in ref [-1,1]^2 coord.

#   integer, intent(in) :: nphys, i_fv
#   real(kind=real_kind), intent(out) :: a(2)

#   integer :: i

#   do i = 0,1
#      a(i+1) = two*(real(i_fv+i-1, real_kind)/real(nphys, real_kind)) - one
#   end do
# end subroutine gfr_f_ref_edges



#   subroutine gfr_init_Dmap(elem, gfr)
#   use control_mod, only: cubed_sphere_map
#   use cube_mod, only: Dmap, ref2sphere
#   use planar_mod, only: plane_Dmap
#   use coordinate_systems_mod, only: cartesian3D_t, change_coordinates

#   type (element_t), intent(in) :: elem(:)
#   type (GllFvRemap_t), intent(inout) :: gfr

#   type (cartesian3D_t) :: sphere
#   real(kind=real_kind) :: wrk(2,2), det, a, b
#   integer :: ie, nf, nf2, i, j, k

#   nf = gfr%nphys
#   nf2 = nf*nf
  
#   ! Jacobian matrices to map a vector between reference element and sphere.
#   do ie = 1,nelemd
#      do j = 1,nf
#         do i = 1,nf
#            if (gfr%is_planar .or. cubed_sphere_map == 0) then
#               call gfr_f_ref_center(nf, i, a)
#               call gfr_f_ref_center(nf, j, b)                
#            else
#               call gfr_f_get_cartesian3d(ie, i, j, sphere)
#               call sphere2ref(elem(ie)%corners3D, sphere, a, b)
#            end if

#            if (gfr%is_planar) then
#               call plane_Dmap(wrk, a, b, elem(ie)%corners3D, cubed_sphere_map, &
#                    elem(ie)%cartp, elem(ie)%facenum)
#            else
#               call       Dmap(wrk, a, b, elem(ie)%corners3D, cubed_sphere_map, &
#                    elem(ie)%cartp, elem(ie)%facenum)
#            end if

#            det = wrk(1,1)*wrk(2,2) - wrk(1,2)*wrk(2,1)

#            ! Make det(D) = fv_metdet. The two should be equal, and fv_metdet
#            ! must be consistent with spherep. Thus, D must be adjusted by a
#            ! scalar.
#            k = i + (j-1)*nf
#            wrk = wrk*sqrt(gfr%fv_metdet(k,ie)/abs(det))
#            det = gfr%fv_metdet(k,ie)

#            gfr%D_f(k,:,:,ie) = wrk

#            gfr%Dinv_f(k,1,1,ie) =  wrk(2,2)/det
#            gfr%Dinv_f(k,1,2,ie) = -wrk(1,2)/det
#            gfr%Dinv_f(k,2,1,ie) = -wrk(2,1)/det
#            gfr%Dinv_f(k,2,2,ie) =  wrk(1,1)/det
#         end do
#      end do
#   end do
# end subroutine gfr_init_Dmap


#   ! GLL -> FV (g2f)

# subroutine gfr_g2f_remapd(gfr, gll_metdet, fv_metdet, g, f)
#   ! Core remap operator. Conservative remap on the reference
#   ! element.

#   type (GllFvRemap_t), intent(in) :: gfr
#   real(kind=real_kind), intent(in) :: gll_metdet(:,:), fv_metdet(:), g(:,:)
#   real(kind=real_kind), intent(out) :: f(:)

#   integer :: nf, nf2, gi, gj, k
#   real(kind=real_kind) :: gw(np,np)

#   nf = gfr%nphys
#   nf2 = nf*nf
#   gw = g*gll_metdet
#   do k = 1,nf2
#      f(k) = sum(gfr%g2f_remapd(:,:,k)*gw)/(gfr%w_ff(k)*fv_metdet(k))
#   end do
# end subroutine gfr_g2f_remapd


# subroutine gfr_f2g_remapd(gfr, gll_metdet, fv_metdet, f, g)
#   ! Core remap operator. Conservative remap on the reference
#   ! element.

#   type (GllFvRemap_t), intent(in) :: gfr
#   real(kind=real_kind), intent(in) :: gll_metdet(:,:), fv_metdet(:), f(:)
#   real(kind=real_kind), intent(out) :: g(:,:)

#   integer :: nf, nf2, gi, gj, fi, fj
#   real(kind=real_kind) :: wrk(np*np)

#   nf = gfr%nphys
#   nf2 = nf*nf
#   wrk(:nf2) = f(:nf2)*fv_metdet(:nf2)
#   do gj = 1,np
#      do gi = 1,np
#         g(gi,gj) = sum(gfr%f2g_remapd(:nf2,gi,gj)*wrk(:nf2))/ &
#              gll_metdet(gi,gj)
#      end do
#   end do
# end subroutine gfr_f2g_remapd

# subroutine apply_interp(interp, np, npi, x, y)
#   ! Apply the npi -> np interpolation matrix 'interp' to x to get y.

#   real(kind=real_kind), intent(in) :: interp(:,:,:,:), x(:,:)
#   integer, intent(in) :: np, npi
#   real(kind=real_kind), intent(out) :: y(:,:)

#   integer :: gi, gj, fi, fj, info
#   real(kind=real_kind) :: accum

#   do fj = 1,np
#      do fi = 1,np
#         accum = zero
#         do gj = 1,npi
#            do gi = 1,npi
#               accum = accum + gfr%interp(gi,gj,fi,fj)*x(gi,gj)
#            end do
#         end do
#         y(fi,fj) = accum
#      end do
#   end do
# end subroutine apply_interp


