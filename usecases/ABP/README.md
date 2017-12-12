Archiv des Bistums Passau (ABP) usecase

*** For DAS 2018 resources, see in resources/DAS_2018 ***


see [this page](TranskribusDU_SPM) for an example using Sequential Pattern Mining

Full workflow for Information Extraction:

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --docid=27734

  python ../../../Local/TranskribusDU/src/tasks/performCVLLA.py --form -i trnskrbs_5400/col/27734.mpxml --coldir=trnskrbs_5400 --docid=27734

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/TranskribusDU_transcriptUploader.py trnskrbs_5400 5400 27734 --nodu

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/do_analyzeLayoutNew.py 5400 --docid=27734/1-151

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --docid=27734 --force --noimage

  python ../../../Local/TranskribusDU/src/tasks/DU_ABPTable_T.py modelMultiType/ TableRowOld_0.1 --run=trnskrbs_5400 

  python ../../../Local/TranskribusDU/src/xml_formats/Page2DS.py --pattern=trnskrbs_5400/col/27734_du.mpxml -o trnskrbs_5400/xml/27734.ds_xml --docid=27734	

  python ../../../../Local/TranskribusDU/src/tasks/rowDetection.py -i trnskrbs_5400/xml/27734.ds_xml  -o trnskrbs_5400/out/27734.ds_xml --docid=27734

  python ../../../Local/TranskribusDU/src/xml_formats/DS2PageXml.py  -i trnskrbs_5400/out/27734.ds_xml -o trnskrbs_5400/col/27734_du.mpxml  --mul

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/TranskribusDU_transcriptUploader.py trnskrbs_5400 5400 27734 

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/do_htrRnn.py 805 DU_ABP_full.dict 5400 --docid=27734/1-151

  python ../../../Local/TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --docid=27734 --force --noimage

  python ../../../../Local/TranskribusDU/usecases/ABP/src/ABP_IE.py --coldir=trnskrbs_5400 --docid=27734 --modelDir=IEdata/model --modelName=model1_h32_nbf1024_epoch128_batch1000  -i trnskrbs_5400/out/27734.ds_xml --usetem
