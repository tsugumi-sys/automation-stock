import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'UzQC6Oh3W3rALdiXFnkEkFf14cxG3lLFNLhkYE22Vlm'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

def main(data):
    symbols = ['A', 'AA', 'AAL', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAWW', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCL', 'ABCM', 'ABEV', 'ABG', 'ABM', 'ABMD', 'ABNB', 'ABT', 'ABTX', 'ABUS', 'ACA', 'ACAD', 'ACB', 'ACBI', 'ACCD', 'ACCO', 'ACEL', 'ACGL', 'ACHC', 'ACI', 'ACIA', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACOR', 'ACTG', 'ACWI', 'ADAG', 'ADAP', 'ADBE', 'ADES', 'ADI', 'ADM', 'ADMA', 'ADMP', 'ADMS', 'ADNT', 'ADP', 'ADPT', 'ADS', 'ADSK', 'ADT', 'ADTN', 'ADUS', 'ADVM', 'AEE', 'AEG', 'AEGN', 'AEIS', 'AEL', 'AEM', 'AEO', 'AEP', 'AER', 'AERI', 'AES', 'AET', 'AEYE', 'AFG', 'AFI', 'AFIB', 'AFK', 'AFL', 'AFRM', 'AFSI', 'AFYA', 'AG', 'AGCO', 'AGEN', 'AGFS', 'AGFY', 'AGG', 'AGGY', 'AGI', 'AGIO', 'AGLE', 'AGM', 'AGO', 'AGR', 'AGS', 'AGTC', 'AGX', 'AGYS', 'AHCO', 'AI', 'AIG', 'AIH', 'AIMC', 'AIN', 'AIQ', 'AIR', 'AIRG', 'AIT', 'AIZ', 'AJG', 'AJRD', 'AKAM', 'AKBA', 'AKRO', 'AKTS', 'AKUS', 'AL', 'ALB', 'ALBO', 'ALC', 'ALCO', 'ALDX', 'ALE', 'ALEC', 'ALG', 'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALIM', 'ALK', 'ALKS', 'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLY', 'ALNY', 'ALOT', 'ALPN', 'ALRM', 'ALRS', 'ALSN', 'ALT', 'ALTA', 'ALTO', 'ALTR', 'ALTY', 'ALV', 'ALVR', 'ALXN', 'ALXO', 'AM', 'AMAL', 'AMAT', 'AMBA', 'AMBC', 'AMC', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMEH', 'AMG', 'AMGN', 'AMK', 'AMKR', 'AMN', 'AMNB', 'AMOT', 'AMP', 'AMPH', 'AMR', 'AMRC', 'AMRN', 'AMRS', 'AMRX', 'AMSC', 'AMSF', 'AMSWA', 'AMTB', 'AMTD', 'AMTI', 'AMTX', 'AMWD', 'AMWL', 'AMX', 'AMZN', 'AN', 'ANAB', 'ANAT', 'ANDE', 'ANDV', 'ANET', 'ANF', 'ANGI', 'ANGO', 'ANIK', 'ANIP', 'ANNX', 'ANSS', 'ANTM', 'AON', 'AOS', 'AOSL', 'AOUT', 'AP', 'APA', 'APAM', 'APD', 'APDN', 'APEI', 'APH', 'APHA', 'API', 'APLS', 'APLT', 'APOG', 'APPF', 'APPN', 'APPS', 'APR', 'APRE', 'APRN', 'APTO', 'APTV', 'APVO', 'APYX', 'AQB', 'AQN', 'AQST', 'AQUA', 'AR', 'ARA', 'ARAV', 'ARAY', 'ARCB', 'ARCC', 'ARCH', 'ARCO', 'ARCT', 'ARD', 'ARDX', 'ARES', 'ARGO', 'ARGX', 'ARLO', 'ARMK', 'ARNA', 'ARNC', 'AROC', 'AROW', 'ARQT', 'ARRS', 'ARRY', 'ARTNA', 'ARVN', 'ARW', 'ARWR', 'ASAN', 'ASB', 'ASGN', 'ASH', 'ASIX', 'ASMB', 'ASML', 'ASO', 'ASPS', 'ASPU', 'ASTE', 'ASV', 'ASX', 'ASYS', 'ATC', 'ATEC', 'ATEN', 'ATEX', 'ATGE', 'ATH', 'ATHA', 'ATHM', 'ATHN', 'ATHX', 'ATI', 'ATKR', 'ATLO', 'ATNI', 'ATNX', 'ATO', 'ATOS', 'ATR', 'ATRA', 'ATRC', 'ATRO', 'ATRS', 'ATSG', 'ATUS', 'ATV', 'ATVI', 'AU', 'AUB', 'AUDC', 'AUPH', 'AUTL', 'AUTO', 'AUY', 'AVAV', 'AVD', 'AVEO', 'AVG', 'AVGO', 'AVID', 'AVIR', 'AVLR', 'AVNS', 'AVNW', 'AVO', 'AVRO', 'AVT', 'AVTR', 'AVXL', 'AVY', 'AVYA', 'AWI', 'AWK', 'AWR', 'AWRE', 'AX', 'AXDX', 'AXGN', 'AXGT', 'AXL', 'AXLA', 'AXNX', 'AXON', 'AXP', 'AXS', 'AXSM', 'AXTA', 'AXTI', 'AYI', 'AYLA', 'AYR', 'AYRO', 'AYX', 'AZEK', 'AZN', 'AZO', 'AZPN', 'AZRE', 'AZYO', 'AZZ', 'B', 'BA', 'BABA', 'BABY', 'BAC', 'BAH', 'BAK', 'BAL', 'BALY', 'BAM', 'BANC', 'BAND', 'BANF', 'BANR', 'BAP', 'BAX', 'BB', 'BBBY', 'BBD', 'BBDC', 'BBH', 'BBI', 'BBIO', 'BBL', 'BBQ', 'BBSI', 'BBVA', 'BBW', 'BBY', 'BC', 'BCAB', 'BCBP', 'BCC', 'BCE', 'BCEI', 'BCEL', 'BCLI', 'BCM', 'BCML', 'BCO', 'BCOV', 'BCPC', 'BCRX', 'BCS', 'BDC', 'BDGE', 'BDSI', 'BDSX', 'BDTX', 'BDX', 'BE', 'BEAM', 'BEAT', 'BECN', 'BEEM', 'BEKE', 'BEN', 'BEPC', 'BERY', 'BEST', 'BF.A', 'BF.B', 'BFAM', 'BFC', 'BFIN', 'BFRA', 'BFST', 'BFYT', 'BG', 'BGCP', 'BGFV', 'BGNE', 'BGS', 'BGSF', 'BH', 'BHC', 'BHE', 'BHF', 'BHLB', 'BHP', 'BHVN', 'BID', 'BIDU', 'BIG', 'BIGC', 'BIIB', 'BILI', 'BILL', 'BIO', 'BIPC', 'BITA', 'BIV', 'BIVV', 'BJ', 'BJRI', 'BK', 'BKD', 'BKE', 'BKH', 'BKI', 'BKNG', 'BKU', 'BL', 'BLBD', 'BLCM', 'BLCT', 'BLD', 'BLDP', 'BLDR', 'BLFS', 'BLI', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLNK', 'BLPH', 'BLUE', 'BLV', 'BLX', 'BMA', 'BMBL', 'BMCH', 'BMI', 'BMO', 'BMRC', 'BMRN', 'BMTC', 'BMY', 'BND', 'BNDX', 'BNFT', 'BNGO', 'BNR', 'BNS', 'BNTX', 'BOCH', 'BOH', 'BOKF', 'BOLT', 'BOMN', 'BOOM', 'BOOT', 'BOTZ', 'BOX', 'BP', 'BPFH', 'BPMC', 'BPTH', 'BQ', 'BR', 'BRBR', 'BRC', 'BRCD', 'BRF', 'BRFS', 'BRID', 'BRK.B', 'BRKL', 'BRKR', 'BRKS', 'BRO', 'BRP', 'BRY', 'BSBR', 'BSET', 'BSGM', 'BSIG', 'BSMX', 'BSRR', 'BSV', 'BSX', 'BSY', 'BTAI', 'BTBT', 'BTI', 'BTU', 'BUD', 'BUG', 'BURL', 'BUSE', 'BV', 'BVS', 'BWA', 'BWB', 'BWEN', 'BWFG', 'BWX', 'BWXT', 'BX', 'BXC', 'BXG', 'BXRX', 'BXS', 'BY', 'BYD', 'BYND', 'BYSI', 'BZH', 'BZUN', 'C', 'CAB', 'CABA', 'CABO', 'CAC', 'CACC', 'CACI', 'CADE', 'CAG', 'CAH', 'CAI', 'CAKE', 'CALA', 'CALM', 'CALT', 'CALX', 'CAMP', 'CAN', 'CAPR', 'CAR', 'CARA', 'CARE', 'CARG', 'CARR', 'CARS', 'CASA', 'CASH', 'CASI', 'CASS', 'CASY', 'CAT', 'CATB', 'CATC', 'CATM', 'CATO', 'CATY', 'CB', 'CBAT', 'CBB', 'CBIO', 'CBPO', 'CBRE', 'CBRL', 'CBS', 'CBSH', 'CBT', 'CBTX', 'CBU', 'CBZ', 'CC', 'CCB', 'CCBG', 'CCCC', 'CCE', 'CCJ', 'CCK', 'CCL', 'CCNE', 'CCOI', 'CCRN', 'CCS', 'CCXI', 'CD', 'CDAK', 'CDAY', 'CDE', 'CDK', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDTX', 'CDW', 'CDXS', 'CDZI', 'CE', 'CECE', 'CECO', 'CEIX', 'CELH', 'CEMI', 'CENT', 'CENTA', 'CENX', 'CEO', 'CERN', 'CERS', 'CERT', 'CETX', 'CEVA', 'CF', 'CFB', 'CFFI', 'CFFN', 'CFG', 'CFR', 'CFRX', 'CFX', 'CGC', 'CGEN', 'CGIX', 'CGNT', 'CGNX', 'CHA', 'CHAD', 'CHAU', 'CHCO', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHFS', 'CHGG', 'CHH', 'CHKP', 'CHL', 'CHMA', 'CHNG', 'CHRS', 'CHRW', 'CHT', 'CHTR', 'CHU', 'CHUY', 'CHWY', 'CHX', 'CI', 'CIA', 'CIEN', 'CIG', 'CINF', 'CIR', 'CIT', 'CIVB', 'CKH', 'CL', 'CLAR', 'CLB', 'CLBK', 'CLCT', 'CLDR', 'CLDX', 'CLF', 'CLFD', 'CLGX', 'CLH', 'CLIR', 'CLNE', 'CLOU', 'CLPS', 'CLR', 'CLSK', 'CLSN', 'CLVS', 'CLVT', 'CLW', 'CLX', 'CLXT', 'CM', 'CMA', 'CMBM', 'CMC', 'CMCM', 'CMCO', 'CMCSA', 'CMD', 'CME', 'CMG', 'CMI', 'CMLS', 'CMP', 'CMPI', 'CMPS', 'CMRX', 'CMS', 'CMTL', 'CNA', 'CNC', 'CNCE', 'CNDT', 'CNHI', 'CNI', 'CNK', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNQ', 'CNR', 'CNS', 'CNSL', 'CNST', 'CNTG', 'CNTY', 'CNX', 'CNXC', 'CNXN', 'CNXT', 'COF', 'COG', 'COHR', 'COHU', 'COKE', 'COLB', 'COLL', 'COLM', 'COMM', 'CONN', 'COO', 'COP', 'CORE', 'CORT', 'COST', 'COT', 'COTY', 'COUP', 'COVS', 'COW', 'COWN', 'CP', 'CPA', 'CPB', 'CPE', 'CPF', 'CPIX', 'CPK', 'CPRI', 'CPRT', 'CPRX', 'CPS', 'CPSI', 'CPST', 'CR', 'CRAI', 'CRBP', 'CRC', 'CREE', 'CRH', 'CRI', 'CRK', 'CRL', 'CRM', 'CRMT', 'CRNC', 'CRNX', 'CRON', 'CROX', 'CRS', 'CRSP', 'CRSR', 'CRTX', 'CRUS', 'CRVL', 'CRVS', 'CRWD', 'CRWS', 'CRY', 'CS', 'CSBR', 'CSCO', 'CSGP', 'CSGS', 'CSII', 'CSIQ', 'CSL', 'CSOD', 'CSPR', 'CSSE', 'CSTL', 'CSV', 'CSWI', 'CSX', 'CTAS', 'CTB', 'CTBI', 'CTEC', 'CTG', 'CTIC', 'CTLT', 'CTMX', 'CTRN', 'CTRP', 'CTS', 'CTSH', 'CTSO', 'CTVA', 'CTXS', 'CUB', 'CUBI', 'CUE', 'CUK', 'CULP', 'CURE', 'CURO', 'CUTR', 'CVA', 'CVAC', 'CVBF', 'CVCO', 'CVCY', 'CVE', 'CVET', 'CVGW', 'CVI', 'CVLG', 'CVLT', 'CVNA', 'CVS', 'CVX', 'CW', 'CWB', 'CWBC', 'CWCO', 'CWEB', 'CWEN', 'CWH', 'CWI', 'CWK', 'CWST', 'CWT', 'CX', 'CXO', 'CXSE', 'CYBE', 'CYBR', 'CYCC', 'CYCN', 'CYD', 'CYH', 'CYRX', 'CYTK', 'CZNC', 'CZR', 'CZWI', 'CZZ', 'D', 'DAC', 'DADA', 'DAIO', 'DAKT', 'DAL', 'DAN', 'DAO', 'DAR', 'DASH', 'DAVA', 'DB', 'DBA', 'DBC', 'DBD', 'DBX', 'DCI', 'DCO', 'DCOM', 'DCPH', 'DCT', 'DD', 'DDD', 'DDOG', 'DDS', 'DE', 'DECK', 'DELL', 'DEM', 'DENN', 'DEO', 'DES', 'DESP', 'DEW', 'DFE', 'DFH', 'DFIN', 'DFJ', 'DFS', 'DG', 'DGICA', 'DGII', 'DGLY', 'DGRE', 'DGRS', 'DGRW', 'DGS', 'DGX', 'DHI', 'DHIL', 'DHR', 'DHS', 'DHT', 'DIA', 'DIN', 'DIOD', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DIV', 'DJP', 'DK', 'DKNG', 'DKS', 'DLB', 'DLN', 'DLTR', 'DLX', 'DMRC', 'DNB', 'DNK', 'DNLI', 'DNOW', 'DOCU', 'DOMO', 'DON', 'DOOR', 'DORM', 'DOV', 'DOW', 'DOX', 'DOYU', 'DPZ', 'DQ', 'DRD', 'DRI', 'DRIO', 'DRIP', 'DRN', 'DRNA', 'DRQ', 'DRV', 'DRVN', 'DS', 'DSGX', 'DSKE', 'DSP', 'DSW', 'DT', 'DTE', 'DTEA', 'DTIL', 'DUK', 'DUO', 'DUST', 'DVA', 'DVAX', 'DVN', 'DVY', 'DXC', 'DXCM', 'DXJ', 'DXJS', 'DXPE', 'DY', 'DZSI', 'EA', 'EAF', 'EAR', 'EAT', 'EB', 'EBAY', 'EBF', 'EBIX', 'EBIZ', 'EBND', 'EBON', 'EBS', 'EBSB', 'EBTC', 'ECA', 'ECHO', 'ECL', 'ECOL', 'ECOM', 'ECPG', 'ED', 'EDC', 'EDIT', 'EDOC', 'EDSA', 'EDU', 'EDUC', 'EDV', 'EDZ', 'EEFT', 'EEM', 'EEMS', 'EEX', 'EFA', 'EFSC', 'EFX', 'EGAN', 'EGBN', 'EGHT', 'EGL', 'EGO', 'EGOV', 'EGRX', 'EH', 'EHC', 'EHTH', 'EIDO', 'EIDX', 'EIG', 'EIGI', 'EIGR', 'EIX', 'EKSO', 'EL', 'ELAN', 'ELD', 'ELF', 'ELY', 'ELYS', 'EMB', 'EME', 'EML', 'EMLC', 'EMN', 'EMR', 'ENB', 'ENDP', 'ENLV', 'ENOB', 'ENPH', 'ENR', 'ENS', 'ENSG', 'ENTA', 'ENTG', 'ENV', 'ENVA', 'EOG', 'EOLS', 'EPAC', 'EPAM', 'EPAY', 'EPC', 'EPHE', 'EPI', 'EPIX', 'EPOL', 'EPP', 'EPZM', 'EQH', 'EQNR', 'EQT', 'ERF', 'ERIC', 'ERIE', 'ERII', 'ERJ', 'ERUS', 'ERX', 'ERY', 'ES', 'ESCA', 'ESE', 'ESGC', 'ESGR', 'ESLT', 'ESNT', 'ESPR', 'ESSA', 'ESTC', 'ETFC', 'ETH', 'ETM', 'ETN', 'ETNB', 'ETR', 'ETRN', 'ETSY', 'EUDG', 'EURN', 'EV', 'EVBG', 'EVER', 'EVFM', 'EVGN', 'EVH', 'EVHC', 'EVOP', 'EVR', 'EVRG', 'EVRI', 'EVTC', 'EW', 'EWBC', 'EWG', 'EWJ', 'EWM', 'EWS', 'EWT', 'EWW', 'EWY', 'EWZ', 'EXAS', 'EXC', 'EXEL', 'EXI', 'EXK', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI', 'EXPO', 'EXPR', 'EXTR', 'EYE', 'EZA', 'EZPW', 'F', 'FAF', 'FANG', 'FANH', 'FARM', 'FARO', 'FAS', 'FAST', 'FATE', 'FAZ', 'FB', 'FBC', 'FBHS', 'FBIZ', 'FBK', 'FBM', 'FBMS', 'FBNC', 'FBP', 'FBR', 'FC', 'FCAP', 'FCBC', 'FCBP', 'FCCY', 'FCEL', 'FCF', 'FCFS', 'FCN', 'FCX', 'FDBC', 'FDN', 'FDP', 'FDS', 'FDX', 'FE', 'FEIM', 'FELE', 'FEYE', 'FEZ', 'FF', 'FFBC', 'FFG', 'FFIC', 'FFIN', 'FFIV', 'FFWM', 'FG', 'FGEN', 'FHB', 'FHI', 'FHN', 'FHTX', 'FIBK', 'FICO', 'FINX', 'FIS', 'FISI', 'FISV', 'FIT', 'FITB', 'FIVE', 'FIVN', 'FIX', 'FIXX', 'FIZZ', 'FL', 'FLDM', 'FLEX', 'FLGT', 'FLIC', 'FLIR', 'FLMN', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT', 'FLWS', 'FLXN', 'FLXS', 'FM', 'FMAO', 'FMBH', 'FMBI', 'FMC', 'FMNB', 'FMS', 'FMTX', 'FMX', 'FN', 'FNB', 'FND', 'FNHC', 'FNKO', 'FNLC', 'FNV', 'FOCS', 'FOE', 'FOLD', 'FOMX', 'FONR', 'FOR', 'FORM', 'FORR', 'FOSL', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FPRX', 'FPX', 'FRAC', 'FRBA', 'FRC', 'FREQ', 'FRG', 'FRGI', 'FRHC', 'FRLN', 'FRME', 'FRO', 'FROG', 'FRPH', 'FRPT', 'FRSX', 'FRTA', 'FSBW', 'FSFG', 'FSLR', 'FSLY', 'FSM', 'FSR', 'FSS', 'FSTR', 'FTCH', 'FTDR', 'FTI', 'FTNT', 'FTS', 'FTV', 'FUBO', 'FUL', 'FULC', 'FULT', 'FUSN', 'FUTU', 'FUV', 'FVCB', 'FVD', 'FVE', 'FVRR', 'FWRD', 'FXI', 'GABC', 'GAL', 'GAN', 'GATO', 'GATX', 'GBCI', 'GBIO', 'GBL', 'GBLI', 'GBT', 'GBX', 'GCI', 'GCO', 'GCP', 'GD', 'GDDY', 'GDEN', 'GDOT', 'GDRX', 'GDS', 'GDX', 'GDXJ', 'GE', 'GEF', 'GEG', 'GEOS', 'GERN', 'GES', 'GFF', 'GFI', 'GFL', 'GFN', 'GGB', 'GGG', 'GH', 'GHC', 'GHG', 'GHL', 'GHLD', 'GHM', 'GIB', 'GIFI', 'GIGM', 'GIII', 'GIL', 'GILD', 'GINN', 'GIS', 'GKOS', 'GLCN', 'GLD', 'GLDD', 'GLDM', 'GLIN', 'GLMD', 'GLNG', 'GLOB', 'GLOG', 'GLRE', 'GLT', 'GLUU', 'GLW', 'GLYC', 'GM', 'GMAB', 'GMBL', 'GMDA', 'GME', 'GMED', 'GMF', 'GMS', 'GNE', 'GNLN', 'GNMK', 'GNOM', 'GNPX', 'GNRC', 'GNTX', 'GNTY', 'GNUS', 'GNW', 'GO', 'GOCO', 'GOGO', 'GOL', 'GOLD', 'GOLF', 'GOOG', 'GOOGL', 'GOOS', 'GOSS', 'GPC', 'GPI', 'GPK', 'GPN', 'GPRE', 'GPRK', 'GPRO', 'GPS', 'GPX', 'GRA', 'GRAY', 'GRBK', 'GRC', 'GRFS', 'GRIL', 'GRMN', 'GRPN', 'GRTS', 'GRTX', 'GRUB', 'GRVY', 'GRWG', 'GS', 'GSBC', 'GSG', 'GSHD', 'GSK', 'GSKY', 'GSMG', 'GSP', 'GSX', 'GT', 'GTES', 'GTH', 'GTHX', 'GTN', 'GTS', 'GTT', 'GTX', 'GURE', 'GUSH', 'GVA', 'GWB', 'GWGH', 'GWPH', 'GWRE', 'GWRS', 'GWW', 'GXTG', 'H', 'HA', 'HACK', 'HAE', 'HAFC', 'HAIN', 'HAL', 'HALL', 'HALO', 'HAPP', 'HARP', 'HAS', 'HAYN', 'HBAN', 'HBB', 'HBCP', 'HBI', 'HBM', 'HBMD', 'HBNC', 'HBT', 'HCA', 'HCAT', 'HCC', 'HCCI', 'HCI', 'HCKT', 'HCM', 'HCSG', 'HD', 'HDB', 'HDS', 'HDV', 'HE', 'HEAR', 'HEDJ', 'HEES', 'HEI', 'HELE', 'HERO', 'HES']
    content = '\n\nToday`s stock report' + str(dt.date.today())

    #----------------------------------------------------------------------------
    for symbol in symbols:   

        try:
            start=dt.datetime.now()-dt.timedelta(days=150)
            end=dt.datetime.now()
            df = yf.download(symbol, start, end, interval='1d')
            # high in the past 81 days
            df['Highest81'] = df['Adj Close'].rolling(window=81).max()
            # 標準偏差を計算
            short_sma = 10
            df['SMA'+str(short_sma)] = df['Adj Close'].rolling(window=short_sma).mean()
            df['STD'] = df['Adj Close'].rolling(window=25).std()
            df['Standard_deviation_normalization'] = 100 * 2 * df['STD'] / df['SMA'+str(short_sma)]


            highest = df['Highest81'][-1]
            highest_2 = df['Highest81'][-60]
            close = df['Adj Close'][-1]
            std = df['Standard_deviation_normalization'][-1]
            #今日の終値が昔の高値よりかなり大きく、上昇トレンドが形成されている
            if close > 1.15 * highest_2 and std < 5 and close > 0.96 * highest:
                content_x = "\n{} 👍 \n逆指値を入れる値段: ¥{}\n前の終値: ¥{}".format(symbol, round(highest, 5), round(close, 5))
                content += '\n'+content_x
            del df

        except :
            continue
        

    send_line_notify(content)