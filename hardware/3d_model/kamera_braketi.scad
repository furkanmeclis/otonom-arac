// ============================================================
// KAMERA_BRAKETI.SCAD — USB Kamera Braketi  v1.0
// Kapağa ayrı parça olarak basılır, 2× M3 ile monte edilir.
//
// BASKI YÖNÜ: taban (BASE) aşağıda, kol (ARM) yukarıda.
// Kamera silindiri yatay basılır → slicer'da yalnızca iç bore
// altında support gerekebilir (~5–15 dakika ek süre).
//
// MONTAJ:
//   Kapağa: 2× M3 vida (taban alt yüzeyi → kapak üst yüzeyi)
//   Kameraya: 2× M3 set vidası (silindir yan duvar, ekvator)
//   USB kablo: kol merkezi → taban → kapak içi
//
// NOT: Gusset'ler (destek üçgenleri) kol tabanına entegre —
// ayrı vida gerektirmez; taban M3 vidaları ile birlikte tutunur.
// ============================================================

$fn = 48;

// === TABAN ===
BASE_W  = 60;    // genişlik y
BASE_D  = 20;    // derinlik x
BASE_T  = 4;     // kalınlık z

// === KOL ===
ARM_D   = 20;    // derinlik x (kablo kanalı etrafında 5mm duvar)
ARM_W   = 20;    // genişlik y
ARM_H   = 50;    // yükseklik z

// === KAMERA YUVASI ===
CAM_ID  = 46;    // iç çap — Ø42mm kamera + 2mm tolerans her yanda
CAM_OD  = 54;    // dış çap — 4mm et kalınlığı
CAM_LN  = 38;    // silindir uzunluğu
CAM_ANG = 15;    // aşağı bakış açısı (derece)

// === GUSSET (DESTEK ÜÇGENİ) ===
GS      = 15;    // üçgen yüksekliği ve tabanı

// === DİĞER ===
CB      = 10;    // USB kablo kanalı (kol içi kare kesit)
M3D     = 3.4;   // M3 vida deliği çapı

// --- Türetilen değerler ---
arm_x0  = (BASE_D - ARM_D) / 2;   // = 0  (kol taban x kenarına flush)
arm_y0  = (BASE_W - ARM_W) / 2;   // = 20 (kol y merkezde)

// ============================================================
// KAMERA YUVASI MODÜLÜ
// Silindir +X yönüne bakar (araç ilerisi), 15° aşağı eğik.
// Kamera ön açık ucundan (x=CAM_LN) kaydırılarak yerleştirilir.
// 2× M3 set vidası z=0 ekvatorda yan duvara dik geçer.
// ============================================================
module kamera_yuvasi() {
    difference() {
        // Dış silindir (X ekseninde, x = 0 → CAM_LN)
        rotate([0, 90, 0])
            cylinder(d=CAM_OD, h=CAM_LN, $fn=64);

        // İç kamera boşluğu
        rotate([0, 90, 0])
            translate([0, 0, -0.1])
                cylinder(d=CAM_ID, h=CAM_LN + 0.2, $fn=64);

        // 2× M3 set vidası — z=0 ekvator, Y ekseni boyunca
        // Duvar kalınlığı bu noktada 4mm, vida tamamen duvarda
        for (lx = [CAM_LN * 0.3, CAM_LN * 0.7])
            translate([lx, 0, 0])
                rotate([90, 0, 0])
                    cylinder(d=M3D, h=CAM_OD + 2, center=true, $fn=24);
    }
}

// ============================================================
// ANA MODEL
// ============================================================
difference() {
    union() {
        // Taban plaka
        cube([BASE_D, BASE_W, BASE_T]);

        // Dikey kol
        translate([arm_x0, arm_y0, BASE_T])
            cube([ARM_D, ARM_W, ARM_H]);

        // ── Gusset sol: y = arm_y0 yüzünden taban sol kenarına ──
        // Hull: kol yüzü kenarı (yüksek) + taban dış noktası (alçak)
        hull() {
            translate([arm_x0, arm_y0 - 0.01, BASE_T])
                cube([ARM_D, 0.01, GS]);
            translate([arm_x0, arm_y0 - GS, BASE_T])
                cube([ARM_D, 0.01, 0.01]);
        }

        // ── Gusset sağ: y = arm_y0+ARM_W yüzünden taban sağ kenarına ──
        hull() {
            translate([arm_x0, arm_y0 + ARM_W - 0.01, BASE_T])
                cube([ARM_D, 0.01, GS]);
            translate([arm_x0, arm_y0 + ARM_W + GS - 0.01, BASE_T])
                cube([ARM_D, 0.01, 0.01]);
        }

        // Silindirik kamera yuvası — kolun ön yüzünde, 15° aşağı eğik
        // x=BASE_D: kolun ön kenarı → bore kol gövdesine girmez
        translate([BASE_D, BASE_W/2, BASE_T + ARM_H])
            rotate([0, CAM_ANG, 0])
                kamera_yuvasi();
    }

    // ── Taban M3 montaj delikleri (2 adet — kapağa bağlantı) ──
    // Global konumlar: x=200, y=70 ve y=110 (kapak.scad ile hizalı)
    for (dy = [BASE_W/2 - 20, BASE_W/2 + 20])
        translate([BASE_D/2, dy, -0.1])
            cylinder(d=M3D, h=BASE_T + 0.2);

    // ── USB kablo kanalı (dikey, kol merkezi) ──
    translate([BASE_D/2 - CB/2, BASE_W/2 - CB/2, BASE_T - 0.1])
        cube([CB, CB, ARM_H + 2]);

    // ── Kablo yatay bağlantı slotu: kanal → kol ön yüzü (silindir arka açıklığı) ──
    // x=5→20, silindir bore'una L-dönüşü ile bağlanır
    translate([BASE_D/2 - CB/2, BASE_W/2 - CB/2, BASE_T + ARM_H - 4])
        cube([BASE_D/2 + CB/2, CB, 6]);

    // ── Kablo taban çıkış deliği ──
    translate([BASE_D/2 - CB/2, BASE_W/2 - CB/2, -0.1])
        cube([CB, CB, BASE_T + 0.2]);
}
