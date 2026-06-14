// ============================================================
// KAPAK.SCAD — RC Otonom Araç Üst Kapak  v1.0
// Boyut: 210 × 180 × 40 mm | Malzeme: PLA / PETG
//
// BASKI YÖNÜ: düz yüzey (üst) aşağıda, açık alt yukarıda.
// Destek (support) gerekmez.
//
// PARÇA SAYISI: 1 (kapak) + 1 (braket, ayrı dosya)
//
// MONTAJ:
//   Şasiye : 4× M4 vida (köşeler)
//   Brakete: 2× M3 vida (ön, üst yüzey)
//   Servis kapağına: 4× M2 vida (orta, üst yüzey)
// ============================================================

$fn = 48;

// === TEMEL ÖLÇÜLER ===
L  = 210;     // uzunluk x
W  = 180;     // genişlik y
H  = 40;      // yükseklik z
T  = 2.5;     // duvar / üst yüzey kalınlığı
CR = 3;       // köşe yarıçapı

// === M4 ŞASİ MONTAJ DELİKLERİ — 4 köşe ===
M4D  = 4.4;
M4MX = 8;
M4MY = 8;

// === SERVİS KAPAĞI AÇIKLIĞI ===
SC_W = 80;    // genişlik x
SC_D = 60;    // derinlik y
// Konumu: kapak merkezinde
// HC_X = (L-SC_W)/2 = 65, HC_Y = (W-SC_D)/2 = 60

// === M2 SERVİS KAPAĞI VİDA DELİKLERİ — 4 adet ===
// Açıklığın dışında, üst yüzey katı bölgesinde
M2D    = 2.4;
HC_X   = (L - SC_W) / 2;             // = 65
HC_Y   = (W - SC_D) / 2;             // = 60
M2_OX  = 5;                          // x kenar payı
M2_OY  = SC_D / 2 - 15;              // y merkeze göre ±15mm → y=75 ve y=105

// === HAVALANDIRMA SLOTLARI — sol + sağ ===
VS_W  = 3;    // slot genişliği
VS_L  = 20;   // slot uzunluğu
VS_N  = 5;    // slot sayısı
VS_SP = 7;    // slotlar arası boşluk
VS_XM = 12;   // x kenarından mesafe

// === YAN DUVAR KABLO ÇIKIŞLARI ===
CB_W = 10;
CB_H = 5;

// === BRAKET M3 DELİKLERİ — üst yüzey, ön kenara yakın ===
// Braket: x=190–210, y=60–120, merkez x=200, y=90
// M3 delikleri x=200'de, y=70 ve y=110
BR_M3D = 3.4;
BR_X   = L - 10;       // = 200
BR_Y1  = W/2 - 20;     // = 70
BR_Y2  = W/2 + 20;     // = 110

// === BRAKET KABLO GEÇİŞ DELİĞİ — üst yüzey ===
// Braket kablo kanalı x=195–205, y=85–95 altına gelir
BK_CB = 10;
BK_X  = L - 15;            // = 195
BK_Y  = W/2 - BK_CB/2;    // = 85

// ============================================================
module rounded_box(l, w, h, r) {
    hull() {
        for (dx = [r, l-r], dy = [r, w-r])
            translate([dx, dy, 0])
                cylinder(r=r, h=h);
    }
}

// ============================================================
// ANA MODEL
// ============================================================
difference() {
    // Dış kabuk
    rounded_box(L, W, H, CR);

    // İç oyma — z=0'dan başlar (alt açık), z=H-T'de durur (üst yüzey T=2.5mm kalır)
    translate([T, T, 0])
        rounded_box(L - 2*T, W - 2*T, H - T, max(CR - T, 0.5));

    // ── M4 şasi montaj delikleri (4 köşe) ──
    for (dx = [M4MX, L - M4MX], dy = [M4MY, W - M4MY])
        translate([dx, dy, H - T - 0.1])
            cylinder(d=M4D, h=T + 0.2);

    // ── Servis kapağı açıklığı (orta) ──
    translate([HC_X, HC_Y, H - T - 0.1])
        cube([SC_W, SC_D, T + 0.2]);

    // ── M2 servis kapağı vida delikleri (4 adet, açıklık kenarı dışında) ──
    for (hx = [HC_X - M2_OX, HC_X + SC_W + M2_OX],
         hy = [HC_Y + M2_OY, HC_Y + SC_D - M2_OY])
        translate([hx, hy, H - T - 0.1])
            cylinder(d=M2D, h=T + 0.2);

    // ── Havalandırma — sol grup ──
    vy0 = (W - VS_N * (VS_L + VS_SP)) / 2;
    for (i = [0 : VS_N - 1])
        translate([VS_XM, vy0 + i*(VS_L + VS_SP), H - T - 0.1])
            cube([VS_W, VS_L, T + 0.2]);

    // ── Havalandırma — sağ grup ──
    for (i = [0 : VS_N - 1])
        translate([L - VS_XM - VS_W, vy0 + i*(VS_L + VS_SP), H - T - 0.1])
            cube([VS_W, VS_L, T + 0.2]);

    // ── Yan duvar kablo çıkışları (sol + sağ) ──
    translate([-0.1, (W - CB_W)/2, T])
        cube([T + 0.2, CB_W, CB_H]);
    translate([L - T - 0.1, (W - CB_W)/2, T])
        cube([T + 0.2, CB_W, CB_H]);

    // ── Braket M3 montaj delikleri (2 adet, üst yüzey) ──
    for (dy = [BR_Y1, BR_Y2])
        translate([BR_X, dy, H - T - 0.1])
            cylinder(d=BR_M3D, h=T + 0.2);

    // ── Braket USB kablo geçiş deliği (üst yüzey) ──
    translate([BK_X, BK_Y, H - T - 0.1])
        cube([BK_CB, BK_CB, T + 0.2]);
}
