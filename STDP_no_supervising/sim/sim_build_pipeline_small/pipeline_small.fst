$date
	Thu Feb 05 23:03:16 2026
$end
$version
	Icarus Verilog
$end
$timescale
	1ps
$end
$scope module pipeline_small $end
$var wire 1 ! clk $end
$var wire 1 " m_tready $end
$var wire 1 # rst $end
$var wire 1 $ s_stdp_en $end
$var wire 20 % s_tdata [19:0] $end
$var wire 1 & s_tvalid $end
$var wire 512 ' w_flat [511:0] $end
$var parameter 32 ( DELAY_E2I_STEPS $end
$var parameter 32 ) DELAY_IN_STEPS $end
$var parameter 32 * DW_CLIP_FP $end
$var parameter 32 + EXC_E_EXC $end
$var parameter 32 , EXC_E_INH $end
$var parameter 32 - EXC_INIT_VTHR $end
$var parameter 32 . EXC_REFRACT $end
$var parameter 32 / EXC_TAU_M $end
$var parameter 32 0 EXC_TC_THETA $end
$var parameter 32 1 EXC_THETA_MAX $end
$var parameter 32 2 EXC_THETA_PLUS_FP $end
$var parameter 32 3 EXC_VPEAK $end
$var parameter 32 4 EXC_VRESET $end
$var parameter 32 5 EXC_VREST $end
$var parameter 32 6 FP_SCALE $end
$var parameter 32 7 FP_SHIFT $end
$var parameter 32 8 INH_E_EXC $end
$var parameter 32 9 INH_E_INH $end
$var parameter 32 : INH_REFRACT $end
$var parameter 32 ; INH_TAU_M $end
$var parameter 32 < INH_VPEAK $end
$var parameter 32 = INH_VRESET $end
$var parameter 32 > INH_VREST $end
$var parameter 32 ? INH_VTHR $end
$var parameter 32 @ LR_M_FP $end
$var parameter 32 A LR_P_FP $end
$var parameter 32 B NORM_FP $end
$var parameter 32 C N_IN $end
$var parameter 32 D N_NEURONS $end
$var parameter 32 E TD_EXC_STEPS $end
$var parameter 32 F TD_INH_STEPS $end
$var parameter 32 G TD_IN_STEPS $end
$var parameter 32 H TD_X_STEPS $end
$var parameter 32 I TSTEP_W $end
$var parameter 32 J UPDATE_NT $end
$var parameter 32 K WEXC_FP $end
$var parameter 32 L WINH_FP $end
$var parameter 32 M WMAX_FP $end
$var parameter 32 N WMIN_FP $end
$var reg 1 O accept_in $end
$var reg 20 P m_tdata [19:0] $end
$var reg 1 Q m_tvalid $end
$var reg 4 R s_exc_next [3:0] $end
$var reg 4 S s_in_effective [3:0] $end
$var reg 4 T s_in_reg [3:0] $end
$var reg 4 U s_inh_next [3:0] $end
$var reg 1 V s_stdp_effective $end
$var reg 1 W s_stdp_reg $end
$var reg 1 X s_tready $end
$var reg 1 Y state $end
$var reg 1 Z stdp_do_update $end
$var reg 4 [ tcount [3:0] $end
$var reg 16 \ tstep_id_reg [15:0] $end
$var integer 32 ] i [31:0] $end
$var integer 32 ^ j [31:0] $end
$var integer 32 _ t [31:0] $end
$scope function fp_div_round $end
$upscope $end
$scope function fp_mul $end
$upscope $end
$scope begin gen_wflat_i[0] $end
$var parameter 2 ` gi $end
$scope begin gen_wflat_j[0] $end
$var parameter 2 a gj $end
$upscope $end
$scope begin gen_wflat_j[1] $end
$var parameter 2 b gj $end
$upscope $end
$scope begin gen_wflat_j[2] $end
$var parameter 3 c gj $end
$upscope $end
$scope begin gen_wflat_j[3] $end
$var parameter 3 d gj $end
$upscope $end
$upscope $end
$scope begin gen_wflat_i[1] $end
$var parameter 2 e gi $end
$scope begin gen_wflat_j[0] $end
$var parameter 2 f gj $end
$upscope $end
$scope begin gen_wflat_j[1] $end
$var parameter 2 g gj $end
$upscope $end
$scope begin gen_wflat_j[2] $end
$var parameter 3 h gj $end
$upscope $end
$scope begin gen_wflat_j[3] $end
$var parameter 3 i gj $end
$upscope $end
$upscope $end
$scope begin gen_wflat_i[2] $end
$var parameter 3 j gi $end
$scope begin gen_wflat_j[0] $end
$var parameter 2 k gj $end
$upscope $end
$scope begin gen_wflat_j[1] $end
$var parameter 2 l gj $end
$upscope $end
$scope begin gen_wflat_j[2] $end
$var parameter 3 m gj $end
$upscope $end
$scope begin gen_wflat_j[3] $end
$var parameter 3 n gj $end
$upscope $end
$upscope $end
$scope begin gen_wflat_i[3] $end
$var parameter 3 o gi $end
$scope begin gen_wflat_j[0] $end
$var parameter 2 p gj $end
$upscope $end
$scope begin gen_wflat_j[1] $end
$var parameter 2 q gj $end
$upscope $end
$scope begin gen_wflat_j[2] $end
$var parameter 3 r gj $end
$upscope $end
$scope begin gen_wflat_j[3] $end
$var parameter 3 s gj $end
$upscope $end
$upscope $end
$upscope $end
$enddefinitions $end
$comment Show the parameter values. $end
$dumpall
b11 s
b10 r
b1 q
b0 p
b11 o
b11 n
b10 m
b1 l
b0 k
b10 j
b11 i
b10 h
b1 g
b0 f
b1 e
b11 d
b10 c
b1 b
b0 a
b0 `
b0 N
b110011001101 M
b1110000000000000 L
b100100000000000000 K
b1000 J
b10000 I
b10100 H
b1 G
b10 F
b1 E
b100 D
b100 C
b1100110011010 B
b1010001111 A
b111 @
b11111111111111111111111111011000 ?
b11111111111111111111111111000100 >
b11111111111111111111111111010011 =
b10100 <
b1010 ;
b10 :
b11111111111111111111111110101011 9
b0 8
b10000 7
b10000000000000000 6
b11111111111111111111111110111111 5
b11111111111111111111111110111111 4
b10100 3
b110011001101 2
b100011 1
b100110001001011010000000 0
b1100100 /
b101 .
b11111111111111111111111111001100 -
b11111111111111111111111110011100 ,
b0 +
b1000010 *
b101 )
b10 (
$end
#0
$dumpvars
b1000 _
b100 ^
b100 ]
b0 \
b0 [
0Z
0Y
1X
0W
0V
b0 U
b0 T
b0 S
b0 R
0Q
b0 P
0O
b1000010000000000000000000000000010000100000000000000000000000000100001000000000000000000000000001000010000000000000000000000000010000100000000000000000000000000100001000000000000000000000000001000010000000000000000000000000010000100000000000000000000000000100001000000000000000000000000001000010000000000000000000000000010000100000000000000000000000000100001000000000000000000000000001000010000000000000000000000000010000100000000000000000000000000100001000000000000000000000000001000010 '
0&
b0 %
0$
1#
1"
1!
$end
#5000
0!
#10000
0#
b100 ^
b1000 _
b10 ]
1!
#15000
0!
#20000
1!
#25000
0!
#30000
b100 ^
b100 ]
1V
b1 S
1O
1$
1&
b1 %
1!
#35000
0!
#40000
0&
0O
1Y
0X
1Q
b1 [
1W
b1 T
b100 ^
b100 ]
1!
#45000
0!
#50000
1&
b10010 %
b100 ^
b100 ]
0Y
0Q
1!
#55000
0!
#60000
b100 ^
b100 ]
b10 S
1O
1X
1!
#65000
0!
#70000
0&
0O
1Y
0X
1Q
b10000 P
b10 [
b10 T
b1 \
b100 ^
b100 ]
1!
#75000
0!
#80000
1&
b100100 %
b100 ^
b100 ]
0Y
0Q
1!
#85000
0!
#90000
b100 ^
b100 ]
b100 S
1O
1X
1!
#95000
0!
#100000
0&
0O
1Y
0X
1Q
b100000 P
b11 [
b100 T
b10 \
b100 ^
b100 ]
1!
#105000
0!
#110000
1&
b111000 %
b100 ^
b100 ]
0Y
0Q
1!
#115000
0!
#120000
b100 ^
b100 ]
b1000 S
1O
1X
1!
#125000
0!
#130000
0&
0O
1Y
0X
1Q
b110000 P
b100 [
b1000 T
b11 \
b100 ^
b100 ]
1!
#135000
0!
#140000
1&
b1001111 %
b100 ^
b100 ]
0Y
0Q
1!
#145000
0!
#150000
b100 ^
b100 ]
b1111 S
1O
1X
1!
#155000
0!
#160000
0&
0O
1Y
0X
1Q
b1000000 P
b101 [
b1111 T
b100 \
b100 ^
b100 ]
1!
#165000
0!
#170000
b100 ^
b100 ]
0Y
0Q
1!
#170001
b100 ^
b100 ]
0$
b0 %
1#
#175001
0!
#180001
0V
b0 S
b0 [
0W
b0 T
b0 \
b0 P
1X
b100 ^
b1000 _
b100 ]
1!
#185001
0!
#190001
0#
b100 ^
b1000 _
b10 ]
1!
#195001
0!
#200001
1!
#205001
0!
#210001
b100 ^
b100 ]
1V
b1 S
1O
1$
1&
b10100001 %
1!
#215001
0!
#220001
0&
0O
1Y
0X
1Q
b10100000 P
b1 [
1W
b1 T
b1010 \
b100 ^
b100 ]
1!
#225001
0!
#230001
1&
b10110001 %
b100 ^
b100 ]
0Y
0Q
1!
#235001
0!
#240001
0"
b100 ^
b100 ]
1O
1X
1!
#245001
0!
#250001
0&
0O
1Y
0X
1Q
b10110000 P
b10 [
b1011 \
b100 ^
b100 ]
1!
#255001
0!
#260001
1"
1!
#265001
0!
#270001
1&
b11000001 %
b100 ^
b100 ]
0Y
0Q
1!
#275001
0!
#280001
b100 ^
b100 ]
1O
1X
1!
#285001
0!
#290001
0"
0&
0O
1Y
0X
1Q
b11000000 P
b11 [
b1100 \
b100 ^
b100 ]
1!
#295001
0!
#300001
1!
#305001
0!
#310001
1"
1!
#315001
0!
#320001
b100 ^
b100 ]
0Y
0Q
1!
#320002
b100 ^
b100 ]
0$
b0 %
1#
#325002
0!
#330002
0V
b0 S
b0 [
0W
b0 T
b0 \
b0 P
1X
b100 ^
b1000 _
b100 ]
1!
#335002
0!
#340002
0#
b100 ^
b1000 _
b10 ]
1!
#345002
0!
#350002
1!
#355002
0!
#360002
b100 ^
b100 ]
1V
b1 S
1O
1$
1&
b1 %
1!
#365002
0!
#370002
0&
0O
1Y
0X
1Q
b1 [
1W
b1 T
b100 ^
b100 ]
1!
#375002
0!
#380002
1&
b10010 %
b100 ^
b100 ]
0Y
0Q
1!
#385002
0!
#390002
b100 ^
b100 ]
b10 S
1O
1X
1!
#395002
0!
#400002
0&
0O
1Y
0X
1Q
b10000 P
b10 [
b10 T
b1 \
b100 ^
b100 ]
1!
#405002
0!
#410002
1&
b100100 %
b100 ^
b100 ]
0Y
0Q
1!
#415002
0!
#420002
b100 ^
b100 ]
b100 S
1O
1X
1!
#425002
0!
#430002
0&
0O
1Y
0X
1Q
b100000 P
b11 [
b100 T
b10 \
b100 ^
b100 ]
1!
#435002
0!
#440002
1&
b111000 %
b100 ^
b100 ]
0Y
0Q
1!
#445002
0!
#450002
b100 ^
b100 ]
b1000 S
1O
1X
1!
#455002
0!
#460002
0&
0O
1Y
0X
1Q
b110000 P
b100 [
b1000 T
b11 \
b100 ^
b100 ]
1!
#465002
0!
#470002
1&
b1001111 %
b100 ^
b100 ]
0Y
0Q
1!
#475002
0!
#480002
b100 ^
b100 ]
b1111 S
1O
1X
1!
#485002
0!
#490002
0&
0O
1Y
0X
1Q
b1000000 P
b101 [
b1111 T
b100 \
b100 ^
b100 ]
1!
#495002
0!
#500002
1&
b1010110 %
b100 ^
b100 ]
0Y
0Q
1!
#505002
0!
#510002
b100 ^
b100 ]
b110 S
1O
1X
1!
#515002
0!
#520002
0&
0O
1Y
0X
1Q
b1010000 P
b110 [
b110 T
b101 \
b100 ^
b100 ]
1!
#525002
0!
#530002
1&
b1100000 %
b100 ^
b100 ]
0Y
0Q
1!
#535002
0!
#540002
b100 ^
b100 ]
b0 S
1O
1X
1!
#545002
0!
#550002
0&
b1000 _
1Z
0O
1Y
0X
1Q
b1100000 P
b111 [
b0 T
b110 \
b100 ^
b100 ]
1!
#555002
0!
#560002
1&
b1111010 %
b1000 _
b100 ^
b100 ]
1Z
0Y
0Q
1!
#565002
0!
#570002
b1000 _
b100 ^
b100 ]
1Z
b1010 S
1O
1X
1!
#575002
0!
#580002
0&
0Z
0O
1Y
0X
1Q
b1110000 P
b0 [
b0 '
b1010 T
b111 \
b100 ^
b100 ]
1!
#585002
0!
#590002
b100 ^
b100 ]
0Y
0Q
1!
#590003
