// ============================================================
// KAPAK_KENARDA.SCAD — RC Otonom Araç Üst Kapak  v2.0
// Montaj tipi: İç tabla (49×150mm) ön kenarda, genişlikte ortalı
//
// Vida pozisyonları DWG "Araç Üst Tabla" çizimine göre:
//   plaka X: 0 → 150mm  (ön kenara dayalı)
//   plaka Y: (180-49)/2=65.5 → 114.5mm (genişlikte ortalı)
//   Ø3.5mm M3 vida → köşeden 3.5mm içeride
//
// BASKI YÖNÜ: düz yüzey (üst) aşağıda, açık alt yukarıda.
// Destek (support) gerekmez.
// ============================================================

$fn = 48;

// === TEMEL ÖLÇÜLER ===
L  = 210;
W  = 180;
H  = 40;
T  = 2.5;
CR = 3;

// === M3 ŞASİ MONTAJ DELİKLERİ — iç tabla ön kenarda ===
// Plaka 49×150mm, ön kenara (x=0) dayalı, y'de ortalı
M3D  = 3.6;
MH_X1 = 3.5;    // 0 + 3.5 (ön kenar)
MH_X2 = 146.5;  // 0 + 150 - 3.5
MH_Y1 = 69;     // (180-49)/2 + 3.5
MH_Y2 = 112;    // (180-49)/2 + 49 - 3.5

// === SERVİS KAPAĞI AÇIKLIĞI ===
SC_W = 80;
SC_D = 60;
HC_X = (L - SC_W) / 2;
HC_Y = (W - SC_D) / 2;
M2D  = 2.4;
M2_OX = 5;
M2_OY = SC_D / 2 - 15;

// === HAVALANDIRMA ===
VS_W  = 3;
VS_L  = 20;
VS_N  = 5;
VS_SP = 7;
VS_XM = 12;

// === KABLO ÇIKIŞLARI ===
CB_W = 10;
CB_H = 5;

// === BRAKET DELİKLERİ ===
BR_M3D = 3.4;
BR_X   = L - 10;
BR_Y1  = W/2 - 20;
BR_Y2  = W/2 + 20;
BK_CB  = 10;
BK_X   = L - 15;
BK_Y   = W/2 - BK_CB/2;

// ============================================================
module rounded_box(l, w, h, r) {
    hull() {
        for (dx = [r, l-r], dy = [r, w-r])
            translate([dx, dy, 0]) cylinder(r=r, h=h);
    }
}

// ============================================================
difference() {
    rounded_box(L, W, H, CR);
    translate([T, T, 0]) rounded_box(L-2*T, W-2*T, H-T, max(CR-T, 0.5));

    // ── M3 şasi montaj delikleri (iç tabla ön kenarda konuma göre) ──
    for (px = [MH_X1, MH_X2], py = [MH_Y1, MH_Y2])
        translate([px, py, H-T-0.1]) cylinder(d=M3D, h=T+0.2);

    // ── Servis kapağı ──
    translate([HC_X, HC_Y, H-T-0.1]) cube([SC_W, SC_D, T+0.2]);

    // ── M2 servis kapağı vida delikleri ──
    for (hx = [HC_X-M2_OX, HC_X+SC_W+M2_OX],
         hy = [HC_Y+M2_OY, HC_Y+SC_D-M2_OY])
        translate([hx, hy, H-T-0.1]) cylinder(d=M2D, h=T+0.2);

    // ── Havalandırma sol ──
    vy0 = (W - VS_N*(VS_L+VS_SP)) / 2;
    for (i = [0:VS_N-1])
        translate([VS_XM, vy0+i*(VS_L+VS_SP), H-T-0.1]) cube([VS_W, VS_L, T+0.2]);

    // ── Havalandırma sağ ──
    for (i = [0:VS_N-1])
        translate([L-VS_XM-VS_W, vy0+i*(VS_L+VS_SP), H-T-0.1]) cube([VS_W, VS_L, T+0.2]);

    // ── Yan kablo çıkışları ──
    translate([-0.1, (W-CB_W)/2, T]) cube([T+0.2, CB_W, CB_H]);
    translate([L-T-0.1, (W-CB_W)/2, T]) cube([T+0.2, CB_W, CB_H]);

    // ── Braket M3 montaj delikleri ──
    for (dy = [BR_Y1, BR_Y2])
        translate([BR_X, dy, H-T-0.1]) cylinder(d=BR_M3D, h=T+0.2);

    // ── Braket kablo geçiş deliği ──
    translate([BK_X, BK_Y, H-T-0.1]) cube([BK_CB, BK_CB, T+0.2]);
}
