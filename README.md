# LAYA Futures Lab

Lokal işləyən, **yalnız Binance USDT perpetual futures** bazarını analiz edən Python tətbiqi. **LAYA real lokal modeldir**: LONG / SHORT / WAIT / CLOSE / BREAKEVEN və leverage təklifi verir. Bir mərkəzi **$2,000 virtual futures hesabı** var. Real exchange order göndərilməsi implementasiya edilməyib; exchange API key tələb olunmur.

## Windows 11-də başlat

**Python 3.12 (64-bit)** və Git quraşdır. Sonra:

```bat
git clone --branch laya-futures https://github.com/fuadqemberov/crypto.git
cd crypto\crypto-laya
setup.cmd
start.cmd
```

`setup.cmd` əsas Python mühitini və ayrıca native Freqtrade mühitini quraşdırır. İlk run zamanı LAYA modelini Hugging Face-dən yükləyir. Sonrakı run-larda model disk cache-dən açılır. UI: **http://127.0.0.1:7860**. İstifadəçi adı `fuad`, şifrə terminalda çıxır. UI-da **Başlat** bas.

16 GB RAM-lı kompüterdə CPU yolu var; model sürəti cihazdan və batch ölçüsündən asılıdır. CUDA-enabled PyTorch quraşdırılıbsa mövcud CUDA GPU avtomatik seçilir. CPU default 4 torch thread və batch 4-dür. CUDA quruluşu üçün rəsmi PyTorch selector-dan istifadə et; bu layihə GPU driver quraşdırmır.

`setup.cmd` xəta verərsə xəta görünən terminalı yoxla. `start.cmd` terminalını bağlamaq prosesi dayandırır. Python 3.12 tapılmırsa `py -0p` ilə quraşdırılmış Python-ları yoxla.

## Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python install_native.py freqtrade
./start.sh
```

CPU inference Windows/Linux/macOS-da istifadə edilə bilər; bu versiyanın avtomatik sürətləndirici seçimi CUDA/CPU-dur. Freqtrade-in seçilən buraxılışı Python 3.11+ tələb edir; əsas quraşdırma Python 3.12 ilə nəzərdə tutulub.

## Nə skan edilir və nə vaxt LAYA qərar verir?

- `exchangeInfo`-dan bütün `TRADING`, `PERPETUAL`, `quoteAsset=USDT` simvolları avtomatik alınır. Spot, delivery, inverse/COIN-M və USDC kontraktları bu universe-ə daxil deyil.
- Texniki skan bütün bu cütlükləri əhatə edir. Yeni listinqdə 250 bağlanmış 5m candle yoxdursa UI-da warmup xətası görünür; data uydurulmur.
- 5m RSI, EMA9/21/200, MACD histogram, ATR və Bollinger Bands hesablanır. **EMA200 = 200 ədəd 5m candle**, 200 günlük EMA deyil.
- Likvidlik minimumu default $1m/24h, spread maksimumu 15 bps-dir. Bütün simvollar UI-da qalır; filterdən keçənlər AI növbəsinə düşür.
- Namizədlər əvvəllər baxılmamış/ən köhnə analiz prioriteti ilə növbələnir. Yalnız top-5 coinə qapanmır. **Hər coinin hər 30 saniyədə LAYA ilə analiz edildiyi iddia edilmir**: bütün universe-in texniki skanı təxminən 60s dövründə, dərin AI analizi default 4 yeni namizəd/batch ilə gedir. CPU, API limitləri və universe ölçüsü müddəti artıra bilər.
- LAYA state-ə əlavə olaraq 15m/1h istiqamət, funding, mark, ekspert datası və açıq mövqe daxil edilir. Açıq mövqelər hər AI dövründə ayrıca prioritetlə yoxlanır.
- LAYA nəticəsi heç vaxt təsadüfi və ya qayda əsaslı fallback qərarı ilə əvəz edilmir. Model xətasında yeni AI order yoxdur; risk monitoru davam edir.
- Yeni giriş eyni simvolun eyni 5m candle-ında təkrar açılmır. Qısa scalping hədəfi yoxdur: default SL məsafəsi `max(2×ATR, qiymətin 0.8%-i)`, TP isə SL məsafəsinin 2 qatıdır. LAYA tez çıxmağı uyğun görərsə CLOSE verə bilər; minimum hold müddəti məcburi deyil.

## Dinamik leverage

LAYA hər inference-də `1, 2, 3, 5, 10, 20×` variantlarından təklif verir. Risk engine bu təklifi:

1. ATR və stop məsafəsi;
2. mövqeyə ayrılan margin;
3. equity-nin default **0.75%** stop-risk büdcəsi;
4. təxmini liquidation məsafəsi;
5. konfiqurasiya maksimumu

ilə məhdudlaşdırır. X sadəcə siqnal balından hesablanan bir rəqəm deyil. UI **LAYA X**, **Applied X** və azaldılma səbəbini göstərir. X yalnız yeni girişdə tətbiq edilir; mövcud mövqenin leverage-i səssiz dəyişdirilmir.

Default maksimum 20×-dir; `.env` ilə aşağı salmaq olar. 20-dən yuxarı dəyər LAYA-nın mövcud seçimlərini genişləndirmir. Hər mövqe üçün margin default equity-nin maksimum 7%-idir. Mövqe sayı ayrıca 5 ilə məhdud deyil; sərbəst margin və ümumi 5% stop-risk büdcəsi ilə məhdudlaşır. Stop-risk hesablaması gap və funding-in gələcək dəyişməsini zəmanətlə məhdudlaşdıra bilməz.

## Köməkçi framework-lər — real status

| Komponent | Bu layihədə rolu | İşə salma |
|---|---|---|
| LAYA 0.3.21 | Əsas real model: action + X, batch inference | Default, əsas lokal mühit |
| Freqtrade 2026.8 | Native `IStrategy` ilə texniki ekspert siqnalı | `setup.cmd` quraşdırır; `FREQTRADE_PYTHON` seçir |
| Hummingbot | Native `BinancePerpetualDerivative(trading_required=False)` ilə futures order-book bridge | Ayrı quraşdırılmış Hummingbot mühitində əlavə observer |
| Jesse 3.2.3 | Native `jesse.research.backtest` ilə 5m futures texniki baseline testləri | Optional ayrıca Python mühiti və UI research tab |
| LEAN | Native crypto-futures research algorithm və Object Store report export | Optional xarici LEAN/QuantConnect runtime |

**Dörd framework-ün hamısı avtomatik quraşdırılıb eyni anda işləyirmiş kimi göstərilmir.** UI-da native statuslar açıqdır. Freqtrade mühiti yoxdursa texniki indikatorlar daxili baseline ilə hesablanır və provider buna uyğun adlandırılır. Hummingbot yoxdursa ictimai futures REST depth analizi işləyir və **Hummingbot runtime** kimi təqdim edilmir. Jesse/LEAN report-u yoxdursa nəticə uydurulmur.

Jesse və LEAN baseline nəticələri **LAYA qərarlarının backtest-i deyil**. Bu hesabatlar timeframe, bazar, simvol, data tarixləri və eyni `technical.py` fingerprint-i ilə yoxlanır. 30 gündən köhnə data yalnız report kimi saxlanır, AI state-ə əlavə edilmir. Eyni indikator məntiqi istifadə olunsa da framework fill/exit modelləri fərqlidir; nəticələri bir-birinin dəqiq ekvivalenti kimi qəbul etmə.

### Jesse əlavə et

```bat
.venv\Scripts\python.exe install_native.py jesse
start.cmd
```

Linux/macOS: `.venv/bin/python install_native.py jesse`, sonra `./start.sh`.

Research tab real Binance futures **1m** tarixçəsini yükləyir, Jesse onu **5m** strategiyada test edir. İlk 1,250 ədəd 1m bar warmup üçündür. Default 4,000 input bar uzunmüddətli validasiya deyil; yalnız qısa funksional sınaqdır. Jesse baseline 1×, 7% position notional, 1.5% stop, 3% TP istifadə edir. Fee stressi 0.07%/side-dır (0.05% fee + 0.02% cost proxy); bu proxy Jesse-də real latency/order-book slippage modeli yaratmır. Funding bu baseline backtest-inə ayrıca daxil edilməyib. Windows-da Jesse dependency quraşdırması problem yaratsa onun ayrıca WSL/Linux mühitindən alınan report-u UI-ya import etmək olar.

### Hummingbot əlavə et

Mövcud Hummingbot source/conda mühitində bu layihənin `integrations/laya_observer.py` faylını çalışdır. Həmin mühitə `gradio_client>=1.14,<2` lazım olur.

```bash
export LAYA_UI_URL=http://127.0.0.1:7860
export LAYA_UI_PASSWORD='UI terminalında çıxan şifrə'
python /path/to/crypto-laya/integrations/laya_observer.py --symbols BTC-USDT,ETH-USDT,SOL-USDT
```

Observer futures connector-u **trading_required=False** ilə yaradır. API key, buy/sell və real order funksiyası yoxdur. Nümunə seçilmiş simvollar üçün native stream verir; bütün bazarın əsas skanını tətbiq özü davam etdirir. Snapshot vaxtı yerli observation timestamp-dir; exchange sequence-dən exchange-time uydurulmur. Köhnəlmiş paketlər 12 saniyədən sonra etibarsız olur. Connector-un lokal kitabı uzun müddət yeni update almırsa bu zaman nişanı onu müstəqil şəkildə sübut etmir; bunu research datası kimi istifadə et.

### LEAN əlavə et

`integrations/lean_project/` native LEAN layihəsidir. Öz LEAN workspace-inə köçür, lazımi **Binance crypto-futures** datasını və uyğun tarix aralığını konfiqurasiya et, sonra:

```bash
lean backtest lean_project
```

Lokal LEAN CLI Docker və uyğun QuantConnect/LEAN quraşdırması tələb edir; əsas tətbiq Docker tələb etmir. Dataset mövcudluğu/lisenziyası avtomatik təmin edilmir. `laya_lean_report.json` Object Store-da yaradılır; həmin JSON-u UI-nın helper tab-ına import et. LEAN adapteri 1×, 5m consolidated bars, növbəti bar qərar icrası, 0.05%/side fee və 0.02% slippage istifadə edir. Stop/TP bar close-da qiymətləndirilir, intrabar fill ilə eyni deyil. Açıq son mövqe equity-də mark olunur; ayrıca funding modeli əlavə edilməyib. Adapter live mode-u rədd edir.

## Futures paper hesabının mənası

- USDT≈USD qəbul edilir. Model **linear USDT-margined** PnL hesablayır: direction × qty × (exit − entry). Leverage notional-ı və margin-i dəyişdirir; PnL ikinci dəfə leverage-ə vurulmur.
- Giriş ask/bid, çıxış bid/ask üzərindən adverse 0.02% slippage ilə simulyasiya olunur. Komissiya 0.05%/side fərziyyəsidir, şəxsi exchange fee tier-i deyil.
- Real funding history-dəki funding time, rate və mark ilə ödəniş hesablanır, hadisə ikinci dəfə tətbiq edilmir. Long positive funding ödəyir, short alır. Bağlanmış mövqelər üzrə gecikən funding 5 dəqiqə ərzində yenidən yoxlanır; endpoint daha uzun gecikərsə fərq qala bilər.
- Mark price SL/TP və təxmini maintenance trigger-i üçün istifadə olunur. Fill real poll zamanı bid/ask ilə edilir. Poll intervalı təxminən 3s-dir; arada toxunub geri qayıdan qiymət görünməyə bilər.
- **Liquidation təxminidir**: sabit 0.5% maintenance ratio. Binance account-specific leverage/notional tier-ləri, liquidation fee, ADL və insurance fund simulyasiya edilmir. Gap zamanı itki gizlədilmir və süni olaraq margin-ə clamp edilmir; bu, birjanın dəqiq isolated-liquidation emulyatoru deyil.
- Minimum lot/tick, real market impact və exchange order acceptance simulyasiya edilməyib. Heç bir real hesab bağlantısı yoxdur.
- Pause zamanı scanner və risk monitoru işləyir. Tətbiq bağlı olduqda onlar işləmir; restart offline qiymət yolunu replay etmir.

## Cache və performans

- Universe TTL 1 saat; toplu 24h ticker 30s; bütün bazarın bid/ask və mark qiymətləri 2s cache.
- HTTP keep-alive connection pool; bounded 8 scanner worker.
- Candle-lar SQLite + RAM-də: warm start, yalnız son 3 candle ilə incremental refresh; timeframe boundary cache version ilə invalidasiya edilir. Gapped/stale history reject olunur.
- Texniki indikator cache-i son bağlanmış candle ilə açarlanır. Freqtrade native nəticəsi də candle başına cache olunur.
- 15m/1h data yalnız AI namizədləri üçün yüklənir və öz timeframe boundary-sinə qədər saxlanır.
- LAYA eyni checkpoint üzərində real `Router.predict_batch` istifadə edir; action və leverage bir typed question batch-dədir. Qərarlar qiymət/position dəyişdiyindən kor-koranə cache edilmir.
- Binance request-weight büdcəsi və funding endpoint üçün ayrıca limit, 429/418 `Retry-After` cooldown var. Bunlar daha çox thread açıb exchange limitini aşmağın qarşısını alır.
- CPU-da yüzlərlə simvolun hamısına saniyəlik model inference zəmanəti yoxdur. UI faktiki scan müddətini, cache hit/miss və model batch ms-ni göstərir.

`.env.example` → `.env` kopyala. `SCAN_WORKERS`, `LAYA_BATCH_SIZE`, `MIN_VOLUME_USDT`, `MAX_LEVERAGE` dəyişə bilər. `.env`, data, virtual hesab və checkpoint-lər Git-ə daxil edilmir.

## Yoxlama

```bash
python -m unittest discover -v
python app.py --no-model
```

`--no-model` yalnız setup diaqnostikasıdır; həmin rejimdə trade başlatmaq bloklanır. Normal `start.cmd` / `start.sh` LAYA-nı aktiv yükləyir.

Unit testlər leverage risk həddi, funding, PnL, stale quote, stop/gap, pause race, checkpoint, cache single-flight, timeframe invalidation, no-lookahead və report schema sərhədlərini əhatə edir. 527 simvolluq sintetik, şəbəkəsiz skan ölçməsində ilk indikator hesablaması 3.734s, cache-dən ikinci dövr 0.006s olub; bu, canlı bazarda end-to-end sürət vədi deyil. Canlı exchangeInfo sorğusunda 527 aktiv USDT perpetual kontrakt aşkar edilib.

Native Freqtrade və Jesse worker-ləri ayrıca sintetik test datası ilə çağırılıb; bu profit benchmark deyil. Hummingbot və LEAN runtime-ları bu hazırlama mühitində end-to-end işlədilməyib.

LAYA-nın ümumi checkpoint-i bu futures strategiyası üçün fine-tune edilməyib. Model probability-ləri kalibrə edilmiş trading win-rate deyil. Daha çox bot birləşdirmək öz-özünə daha yaxşı gəlir yaratmır; model və strategiyanı uzun müddət paper rejimdə ölçmək lazımdır.

## Mənbələr

- https://github.com/NandhaKishorM/laya
- https://www.freqtrade.io/en/stable/strategy-customization/
- https://docs.jesse.trade/docs/research/backtest.html
- https://github.com/hummingbot/hummingbot
- https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/crypto-futures/requesting-data
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information
