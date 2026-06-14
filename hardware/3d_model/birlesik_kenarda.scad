// ============================================================
// BIRLESİK_KENARDA.SCAD — Montaj Önizlemesi (İç tabla ön kenarda)
// SADECE görsel kontrol. Baskı için kapak_kenarda.scad kullanın.
// F6 = tam render
// ============================================================

$fn = 48;

// ── Kapak parametreleri ──
L=210; W=180; H=40; T=2.5; CR=3;
M3D=3.6; MH_X1=3.5; MH_X2=146.5; MH_Y1=69; MH_Y2=112;
SC_W=80; SC_D=60;
HC_X=(L-SC_W)/2; HC_Y=(W-SC_D)/2;
M2D=2.4; M2_OX=5; M2_OY=SC_D/2-15;
VS_W=3; VS_L=20; VS_N=5; VS_SP=7; VS_XM=12;
CB_W=10; CB_H=5;
BR_M3D=3.4; BR_X=L-10; BR_Y1=W/2-20; BR_Y2=W/2+20;
BK_CB=10; BK_X=L-15; BK_Y=W/2-BK_CB/2;

// ── Braket parametreleri ──
BASE_W=60; BASE_D=20; BASE_T=4;
ARM_D=20; ARM_W=20; ARM_H=50;
CAM_ID=46; CAM_OD=54; CAM_LN=38; CAM_ANG=15;
GS=15; CB=10; BM3D=3.4;
arm_x0=(BASE_D-ARM_D)/2;
arm_y0=(BASE_W-ARM_W)/2;

module rounded_box(l,w,h,r) {
    hull() { for(dx=[r,l-r],dy=[r,w-r]) translate([dx,dy,0]) cylinder(r=r,h=h); }
}

module kapak() {
    vy0=(W-VS_N*(VS_L+VS_SP))/2;
    difference() {
        rounded_box(L,W,H,CR);
        translate([T,T,0]) rounded_box(L-2*T,W-2*T,H-T,max(CR-T,0.5));
        for(px=[MH_X1,MH_X2],py=[MH_Y1,MH_Y2])
            translate([px,py,H-T-0.1]) cylinder(d=M3D,h=T+0.2);
        translate([HC_X,HC_Y,H-T-0.1]) cube([SC_W,SC_D,T+0.2]);
        for(hx=[HC_X-M2_OX,HC_X+SC_W+M2_OX],hy=[HC_Y+M2_OY,HC_Y+SC_D-M2_OY])
            translate([hx,hy,H-T-0.1]) cylinder(d=M2D,h=T+0.2);
        for(i=[0:VS_N-1]) translate([VS_XM,vy0+i*(VS_L+VS_SP),H-T-0.1]) cube([VS_W,VS_L,T+0.2]);
        for(i=[0:VS_N-1]) translate([L-VS_XM-VS_W,vy0+i*(VS_L+VS_SP),H-T-0.1]) cube([VS_W,VS_L,T+0.2]);
        translate([-0.1,(W-CB_W)/2,T]) cube([T+0.2,CB_W,CB_H]);
        translate([L-T-0.1,(W-CB_W)/2,T]) cube([T+0.2,CB_W,CB_H]);
        for(dy=[BR_Y1,BR_Y2]) translate([BR_X,dy,H-T-0.1]) cylinder(d=BR_M3D,h=T+0.2);
        translate([BK_X,BK_Y,H-T-0.1]) cube([BK_CB,BK_CB,T+0.2]);
    }
}

module kamera_yuvasi() {
    difference() {
        rotate([0,90,0]) cylinder(d=CAM_OD,h=CAM_LN,$fn=64);
        rotate([0,90,0]) translate([0,0,-0.1]) cylinder(d=CAM_ID,h=CAM_LN+0.2,$fn=64);
        for(lx=[CAM_LN*0.3,CAM_LN*0.7])
            translate([lx,0,0]) rotate([90,0,0]) cylinder(d=BM3D,h=CAM_OD+2,center=true,$fn=24);
    }
}

module kamera_braketi() {
    difference() {
        union() {
            cube([BASE_D,BASE_W,BASE_T]);
            translate([arm_x0,arm_y0,BASE_T]) cube([ARM_D,ARM_W,ARM_H]);
            hull() {
                translate([arm_x0,arm_y0-0.01,BASE_T]) cube([ARM_D,0.01,GS]);
                translate([arm_x0,arm_y0-GS,BASE_T]) cube([ARM_D,0.01,0.01]);
            }
            hull() {
                translate([arm_x0,arm_y0+ARM_W-0.01,BASE_T]) cube([ARM_D,0.01,GS]);
                translate([arm_x0,arm_y0+ARM_W+GS-0.01,BASE_T]) cube([ARM_D,0.01,0.01]);
            }
            translate([BASE_D,BASE_W/2,BASE_T+ARM_H]) rotate([0,CAM_ANG,0]) kamera_yuvasi();
        }
        for(dy=[BASE_W/2-20,BASE_W/2+20])
            translate([BASE_D/2,dy,-0.1]) cylinder(d=BM3D,h=BASE_T+0.2);
        translate([BASE_D/2-CB/2,BASE_W/2-CB/2,BASE_T-0.1]) cube([CB,CB,ARM_H+2]);
        translate([BASE_D/2-CB/2,BASE_W/2-CB/2,BASE_T+ARM_H-4]) cube([BASE_D/2+CB/2,CB,6]);
        translate([BASE_D/2-CB/2,BASE_W/2-CB/2,-0.1]) cube([CB,CB,BASE_T+0.2]);
    }
}

color("DodgerBlue",0.9) kapak();
color("SlateGray",0.9)
    translate([L-BASE_D,(W-BASE_W)/2,H]) kamera_braketi();
