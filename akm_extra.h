#pragma once
#include <string>
#include <unordered_map>

struct AkmRule {
    std::string fixed8;
    std::string fixed1;
    std::string fixed2;
    std::string fixed3;
    std::string pad_nibble;
    bool repeat_last = false;
};

inline void load_akm_extra_profile(const std::string& name, 
                                   std::unordered_map<std::string, std::string>& customHex, 
                                   std::unordered_map<std::string, AkmRule>& specialRules) {
    
    customHex.clear();
    specialRules.clear();

    // =========================================================
    // 1. akm2-lab-v1
    // =========================================================
    if (name == "akm2-lab-v1") {
        customHex["jocul"] = "dead";
        customHex["ochiuri"] = "face";
        customHex["privire"] = "0e0e";

        specialRules["soarele"] = { "cafebabe", "", "", "", "", false };
        specialRules["norocel"] = { "", "", "", "", "7", false };
        specialRules["paradis"] = { "", "", "", "", "", true }; // repeat_last
    }

    // =========================================================
    // 2. akm2-lab-strict-cs
    // =========================================================
    else if (name == "akm2-lab-strict-cs") {
        // Nu are hex/rules custom, foloseste doar politica de checksum
        // (care in C++ e hardcodata momentan pe V1/root)
        // Lasam gol, va folosi fallback-ul din akm.h (base words)
    }

    // =========================================================
    // 3. akm2-fixed123-pack-v1
    // =========================================================
    else if (name == "akm2-fixed123-pack-v1") {
        // Custom Hex
        customHex["abis"]="CB46"; customHex["acelasi"]="12"; customHex["acoperis"]="123"; customHex["adanc"]="b";
        customHex["adapost"]="34"; customHex["adorare"]="345"; customHex["aer"]="c"; customHex["ajun"]="56";
        customHex["albastru"]="567"; customHex["alb"]="d"; customHex["alge"]="89"; customHex["altar"]="89a";
        customHex["amintire"]="e"; customHex["amurg"]="cd"; customHex["anotimp"]="cde"; customHex["apa"]="f";
        customHex["apele"]="01"; customHex["apus"]="012"; customHex["apuseni"]="02"; customHex["arta"]="234";

        // Special Rules
        specialRules["abis"] = { "", "CB46", "", "", "", false };
        specialRules["adanc"] = { "", "b", "", "", "", false };
        specialRules["amintire"] = { "", "e", "", "", "", false };
        specialRules["bujor"] = { "", "", "", "000", "", false };

        specialRules["acelasi"] = { "", "", "12", "", "", false };
        specialRules["ajun"] = { "", "", "56", "", "", false };
        specialRules["apele"] = { "", "", "f1", "", "", false };
        specialRules["apuseni"] = { "", "", "d2", "", "", false };

        specialRules["acoperis"] = { "", "", "", "25E", "", false };
        specialRules["adorare"] = { "", "", "", "345", "", false };
        specialRules["albastru"] = { "", "", "", "567", "", false };
        specialRules["altar"] = { "", "", "", "89a", "", false };
        specialRules["anotimp"] = { "", "", "", "cde", "", false };
        specialRules["arta"] = { "", "", "", "234", "", false };

        specialRules["ochiuri"] = { "facefeed", "", "", "", "", false };
        specialRules["privire"] = { "0e0e0e0e", "", "", "", "", false };
    }

    // =========================================================
    // 4. akm3-puzzle71
    // =========================================================
    else if (name == "akm3-puzzle71") {
        customHex["abis"]="100"; customHex["acelasi"]="101"; customHex["acoperis"]="102"; customHex["adanc"]="103";
        customHex["adapost"]="104"; customHex["adorare"]="105"; customHex["afectiune"]="106"; customHex["aer"]="107";
        customHex["ajun"]="108"; customHex["albastru"]="109"; customHex["alb"]="10a"; customHex["alge"]="10b";
        customHex["altar"]="10c"; customHex["amintire"]="10d"; customHex["amurg"]="10e"; customHex["anotimp"]="10f";
        customHex["apa"]="110"; customHex["apele"]="111"; customHex["apus"]="112"; customHex["apuseni"]="113";
        customHex["apusor"]="114"; customHex["aroma"]="115"; customHex["artar"]="116"; customHex["arta"]="117";
        customHex["asfintit"]="118"; customHex["asteptare"]="119"; customHex["atingere"]="11a"; customHex["aur"]="11b";
        customHex["aurora"]="11c"; customHex["autumn"]="11d"; customHex["avion"]="11e"; customHex["balta"]="11f";
        customHex["barca"]="120"; customHex["barza"]="121"; customHex["baterie"]="122"; customHex["batran"]="123";
        customHex["bec"]="124"; customHex["bezna"]="125"; customHex["binecuvantare"]="126"; customHex["blandete"]="127";
        customHex["boboc"]="128"; customHex["bogatie"]="129"; customHex["bolta"]="12a"; customHex["brad"]="12b";
        customHex["brat"]="12c"; customHex["bruma"]="12d"; customHex["bucata"]="12e"; customHex["bucurie"]="12f";
        customHex["bujor"]="130"; customHex["burg"]="131"; customHex["camp"]="132"; customHex["campie"]="133";
        customHex["cafea"]="134"; customHex["calator"]="135"; customHex["cald"]="136"; customHex["candela"]="137";
        customHex["caprioara"]="138"; customHex["caramida"]="139"; customHex["carare"]="13a"; customHex["carte"]="13b";
        customHex["catel"]="13c"; customHex["cautare"]="13d"; customHex["casa"]="13e"; customHex["ceas"]="13f";
        customHex["cer"]="140"; customHex["cerb"]="141"; customHex["chip"]="142"; customHex["ciocarlie"]="143";
        customHex["ciutura"]="144"; customHex["clar"]="145"; customHex["clipa"]="146"; customHex["clopot"]="147";
        customHex["coborare"]="148"; customHex["colina"]="149"; customHex["colt"]="14a"; customHex["copac"]="14b";
        customHex["copil"]="14c"; customHex["corabie"]="14d"; customHex["cord"]="14e"; customHex["corn"]="14f";
        customHex["crang"]="150"; customHex["credinta"]="151"; customHex["crestere"]="152"; customHex["crestet"]="153";
        customHex["crin"]="154"; customHex["cuc"]="155"; customHex["cufar"]="156"; customHex["culoare"]="157";
        customHex["culme"]="158"; customHex["curcubeu"]="159"; customHex["curte"]="15a"; customHex["cupru"]="15b";
        customHex["cuvant"]="15c"; customHex["cutie"]="15d"; customHex["daurire"]="15e"; customHex["deal"]="15f";
        customHex["deget"]="160"; customHex["delusor"]="161"; customHex["departare"]="162"; customHex["desert"]="163";
        customHex["dimineata"]="164"; customHex["dor"]="165"; customHex["dorinta"]="166"; customHex["drag"]="167";
        customHex["draga"]="168"; customHex["drum"]="169"; customHex["drumet"]="16a"; customHex["durere"]="16b";
        customHex["duminica"]="16c"; customHex["ecou"]="16d"; customHex["efemer"]="16e"; customHex["elixir"]="16f";
        customHex["emisfera"]="170"; customHex["enigma"]="171"; customHex["eter"]="172"; customHex["eternitate"]="173";
        customHex["fag"]="174"; customHex["fagure"]="175"; customHex["fantana"]="176"; customHex["farmec"]="177";
        customHex["fata"]="178"; customHex["felinar"]="179"; customHex["fenic"]="17a"; customHex["fereastra"]="17b";
        customHex["fericire"]="17c"; customHex["feriga"]="17d"; customHex["fier"]="17e"; customHex["fierar"]="17f";
        customHex["film"]="180"; customHex["fior"]="181"; customHex["flacara"]="182"; customHex["flamura"]="183";
        customHex["floare"]="184"; customHex["fluture"]="185"; customHex["fosnet"]="186"; customHex["fotografie"]="187";
        customHex["frag"]="188"; customHex["frate"]="189"; customHex["frezie"]="18a"; customHex["frig"]="18b";
        customHex["fruct"]="18c"; customHex["frumusete"]="18d"; customHex["frunza"]="18e"; customHex["frunte"]="18f";
        customHex["fulger"]="190"; customHex["furnica"]="191"; customHex["galaxie"]="192"; customHex["galben"]="193";
        customHex["gand"]="194"; customHex["gandire"]="195"; customHex["garoafa"]="196"; customHex["gheata"]="197";
        customHex["ghetar"]="198"; customHex["ghinda"]="199"; customHex["ghiozdan"]="19a"; customHex["glas"]="19b";
        customHex["glorie"]="19c"; customHex["grad"]="19d"; customHex["gradina"]="19e"; customHex["grai"]="19f";
        customHex["granita"]="1a0"; customHex["gust"]="1a1"; customHex["gura"]="1a2"; customHex["har"]="1a3";
        customHex["harfa"]="1a4"; customHex["iarba"]="1a5"; customHex["iarna"]="1a6"; customHex["icoana"]="1a7";
        customHex["implinire"]="1a8"; customHex["inger"]="1a9"; customHex["insula"]="1aa"; customHex["insorire"]="1ab";
        customHex["intindere"]="1ac"; customHex["intuneric"]="1ad"; customHex["inviere"]="1ae"; customHex["iubire"]="1af";
        customHex["iz"]="1b0"; customHex["izvor"]="1b1"; customHex["izvoras"]="1b2"; customHex["joc"]="1b3";
        customHex["jocul"]="1b4"; customHex["lac"]="1b5"; customHex["lacrima"]="1b6"; customHex["laur"]="1b7";
        customHex["lebada"]="1b8"; customHex["legenda"]="1b9"; customHex["lemn"]="1ba"; customHex["leu"]="1bb";
        customHex["libertate"]="1bc"; customHex["linie"]="1bd"; customHex["livada"]="1be"; customHex["loc"]="1bf";
        customHex["luna"]="1c0"; customHex["lumina"]="1c1"; customHex["lume"]="1c2"; customHex["lunca"]="1c3";
        customHex["lup"]="1c4"; customHex["lut"]="1c5"; customHex["manunchi"]="1c6"; customHex["margine"]="1c7";

        specialRules["abis"]={ "","0" }; specialRules["acelasi"]={ "","","01" }; specialRules["acoperis"]={ "","","","102" };
        specialRules["adanc"]={ "","3" }; specialRules["adapost"]={ "","","04" }; specialRules["adorare"]={ "","","","105" };
        specialRules["afectiune"]={ "","6" }; specialRules["aer"]={ "","","07" }; specialRules["ajun"]={ "","","","108" };
        specialRules["albastru"]={ "","9" }; specialRules["alb"]={ "","","0a" }; specialRules["alge"]={ "","","","10b" };
        specialRules["altar"]={ "","c" }; specialRules["amintire"]={ "","","0d" }; specialRules["amurg"]={ "","","","10e" };
        specialRules["anotimp"]={ "","f" }; specialRules["apa"]={ "","","10" }; specialRules["apele"]={ "","","","111" };
        specialRules["apus"]={ "","2" }; specialRules["apuseni"]={ "","","13" }; specialRules["apusor"]={ "","","","114" };
        specialRules["aroma"]={ "","5" }; specialRules["artar"]={ "","","16" }; specialRules["arta"]={ "","","","117" };
        specialRules["asfintit"]={ "","8" }; specialRules["asteptare"]={ "","","19" }; specialRules["atingere"]={ "","","","11a" };
        specialRules["aur"]={ "","b" }; specialRules["aurora"]={ "","","1c" }; specialRules["autumn"]={ "","","","11d" };
        specialRules["avion"]={ "","e" }; specialRules["balta"]={ "","","1f" }; specialRules["barca"]={ "","","","120" };
        specialRules["barza"]={ "","1" }; specialRules["baterie"]={ "","","22" }; specialRules["batran"]={ "","","","123" };
        specialRules["bec"]={ "","4" }; specialRules["bezna"]={ "","","25" }; specialRules["binecuvantare"]={ "","","","126" };
        specialRules["blandete"]={ "","7" }; specialRules["boboc"]={ "","","28" }; specialRules["bogatie"]={ "","","","129" };
        specialRules["bolta"]={ "","a" }; specialRules["brad"]={ "","","2b" }; specialRules["brat"]={ "","","","12c" };
        specialRules["bruma"]={ "","d" }; specialRules["bucata"]={ "","","2e" }; specialRules["bucurie"]={ "","","","12f" };
        specialRules["bujor"]={ "","0" }; specialRules["burg"]={ "","","31" }; specialRules["camp"]={ "","","","132" };
        specialRules["campie"]={ "","3" }; specialRules["cafea"]={ "","","34" }; specialRules["calator"]={ "","","","135" };
        specialRules["cald"]={ "","6" }; specialRules["candela"]={ "","","37" }; specialRules["caprioara"]={ "","","","138" };
        specialRules["caramida"]={ "","9" }; specialRules["carare"]={ "","","3a" }; specialRules["carte"]={ "","","","13b" };
        specialRules["catel"]={ "","c" }; specialRules["cautare"]={ "","","3d" }; specialRules["casa"]={ "","","","13e" };
        specialRules["ceas"]={ "","f" }; specialRules["cer"]={ "","","40" }; specialRules["cerb"]={ "","","","141" };
        specialRules["chip"]={ "","2" }; specialRules["ciocarlie"]={ "","","43" }; specialRules["ciutura"]={ "","","","144" };
        specialRules["clar"]={ "","5" }; specialRules["clipa"]={ "","","46" }; specialRules["clopot"]={ "","","","147" };
        specialRules["coborare"]={ "","8" }; specialRules["colina"]={ "","","49" }; specialRules["colt"]={ "","","","14a" };
        specialRules["copac"]={ "","b" }; specialRules["copil"]={ "","","4c" }; specialRules["corabie"]={ "","","","14d" };
        specialRules["cord"]={ "","e" }; specialRules["corn"]={ "","","4f" }; specialRules["crang"]={ "","","","150" };
        specialRules["credinta"]={ "","1" }; specialRules["crestere"]={ "","","52" }; specialRules["crestet"]={ "","","","153" };
        specialRules["crin"]={ "","4" }; specialRules["cuc"]={ "","","55" }; specialRules["cufar"]={ "","","","156" };
        specialRules["culoare"]={ "","7" }; specialRules["culme"]={ "","","58" }; specialRules["curcubeu"]={ "","","","159" };
        specialRules["curte"]={ "","a" }; specialRules["cupru"]={ "","","5b" }; specialRules["cuvant"]={ "","","","15c" };
        specialRules["cutie"]={ "","d" }; specialRules["daurire"]={ "","","5e" }; specialRules["deal"]={ "","","","15f" };
        specialRules["deget"]={ "","0" }; specialRules["delusor"]={ "","","61" }; specialRules["departare"]={ "","","","162" };
        specialRules["desert"]={ "","3" }; specialRules["dimineata"]={ "","","64" }; specialRules["dor"]={ "","","","165" };
        specialRules["dorinta"]={ "","6" }; specialRules["drag"]={ "","","67" }; specialRules["draga"]={ "","","","168" };
        specialRules["drum"]={ "","9" }; specialRules["drumet"]={ "","","6a" }; specialRules["durere"]={ "","","","16b" };
        specialRules["duminica"]={ "","c" }; specialRules["ecou"]={ "","","6d" }; specialRules["efemer"]={ "","","","16e" };
        specialRules["elixir"]={ "","f" }; specialRules["emisfera"]={ "","","70" }; specialRules["enigma"]={ "","","","171" };
        specialRules["eter"]={ "","2" }; specialRules["eternitate"]={ "","","73" }; specialRules["fag"]={ "","","","174" };
        specialRules["fagure"]={ "","5" }; specialRules["fantana"]={ "","","76" }; specialRules["farmec"]={ "","","","177" };
        specialRules["fata"]={ "","8" }; specialRules["felinar"]={ "","","79" }; specialRules["fenic"]={ "","","","17a" };
        specialRules["fereastra"]={ "","b" }; specialRules["fericire"]={ "","","7c" }; specialRules["feriga"]={ "","","","17d" };
        specialRules["fier"]={ "","e" }; specialRules["fierar"]={ "","","7f" }; specialRules["film"]={ "","","","180" };
        specialRules["fior"]={ "","1" }; specialRules["flacara"]={ "","","82" }; specialRules["flamura"]={ "","","","183" };
        specialRules["floare"]={ "","4" }; specialRules["fluture"]={ "","","85" }; specialRules["fosnet"]={ "","","","186" };
        specialRules["fotografie"]={ "","7" }; specialRules["frag"]={ "","","88" }; specialRules["frate"]={ "","","","189" };
        specialRules["frezie"]={ "","a" }; specialRules["frig"]={ "","","8b" }; specialRules["fruct"]={ "","","","18c" };
        specialRules["frumusete"]={ "","d" }; specialRules["frunza"]={ "","","8e" }; specialRules["frunte"]={ "","","","18f" };
        specialRules["fulger"]={ "","0" }; specialRules["furnica"]={ "","","91" }; specialRules["galaxie"]={ "","","","192" };
        specialRules["galben"]={ "","3" }; specialRules["gand"]={ "","","94" }; specialRules["gandire"]={ "","","","195" };
        specialRules["garoafa"]={ "","6" }; specialRules["gheata"]={ "","","97" }; specialRules["ghetar"]={ "","","","198" };
        specialRules["ghinda"]={ "","9" }; specialRules["ghiozdan"]={ "","","9a" }; specialRules["glas"]={ "","","","19b" };
        specialRules["glorie"]={ "","c" }; specialRules["grad"]={ "","","9d" }; specialRules["gradina"]={ "","","","19e" };
        specialRules["grai"]={ "","f" }; specialRules["granita"]={ "","","a0" }; specialRules["gust"]={ "","","","1a1" };
        specialRules["gura"]={ "","2" }; specialRules["har"]={ "","","a3" }; specialRules["harfa"]={ "","","","1a4" };
        specialRules["iarba"]={ "","5" }; specialRules["iarna"]={ "","","a6" }; specialRules["icoana"]={ "","","","1a7" };
        specialRules["implinire"]={ "","8" }; specialRules["inger"]={ "","","a9" }; specialRules["insula"]={ "","","","1aa" };
        specialRules["insorire"]={ "","b" }; specialRules["intindere"]={ "","","ac" }; specialRules["intuneric"]={ "","","","1ad" };
        specialRules["inviere"]={ "","e" }; specialRules["iubire"]={ "","","af" }; specialRules["iz"]={ "","","","1b0" };
        specialRules["izvor"]={ "","1" }; specialRules["izvoras"]={ "","","b2" }; specialRules["joc"]={ "","","","1b3" };
        specialRules["jocul"]={ "","4" }; specialRules["lac"]={ "","","b5" }; specialRules["lacrima"]={ "","","","1b6" };
        specialRules["laur"]={ "","7" }; specialRules["lebada"]={ "","","b8" }; specialRules["legenda"]={ "","","","1b9" };
        specialRules["lemn"]={ "","a" }; specialRules["leu"]={ "","","bb" }; specialRules["libertate"]={ "","","","1bc" };
        specialRules["linie"]={ "","d" }; specialRules["livada"]={ "","","be" }; specialRules["loc"]={ "","","","1bf" };
        specialRules["luna"]={ "","0" }; specialRules["lumina"]={ "","","c1" }; specialRules["lume"]={ "","","","1c2" };
        specialRules["lunca"]={ "","3" }; specialRules["lup"]={ "","","c4" }; specialRules["lut"]={ "","","","1c5" };
        specialRules["manunchi"]={ "","6" }; customHex["margine"]="1c7"; specialRules["margine"]={ "","","c7" };
    }
}