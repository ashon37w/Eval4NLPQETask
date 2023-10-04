from xml.dom.minidom import  parse
import xml.dom.minidom

def get_sentence(file):
    rst={}
    domTree = parse(file)
    docs = domTree.getElementsByTagName("doc")

    for doc in docs:
        # print("docid",doc.getAttribute('docid'))
        docid=doc.getAttribute('docid')
        rst.setdefault(doc.getAttribute('docid'),{})
        segs = doc.getElementsByTagName('seg')
        for seg in segs:
            #print("id:%d,sentence:%s"%(int(seg.getAttribute('id')),seg.childNodes[0].data))
            rst[docid].setdefault(seg.getAttribute('id'), seg.childNodes[0].data)
    return  rst


en_src=get_sentence(file=r"./dev/sgm/newstest2013-src.es.sgm")
cn_ref=get_sentence(file=r"./dev/sgm/newstest2013-ref.en.sgm")

with open("./esen_es_dev.txt","w+") as en_fp:
    with open("./esen_en_dev.txt","w+") as cn_fp:
        for docid in en_src:
            for id in en_src[docid]:
                if (docid in cn_ref) and (id in cn_ref[docid]):
                    en_fp.writelines(en_src[docid][id]+"\n")
                    cn_fp.writelines(cn_ref[docid][id]+"\n")
