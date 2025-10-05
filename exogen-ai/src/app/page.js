import Header from '@/app/components/header';
import Image from 'next/image';
import bg_planet from '@/app/assets/svg/background_planet.svg';

export default function Home() {
  return (
    <div>
      <Header />
      <div className="flex justify-center items-center relative">
        <span>
          <div className="absolute left-40 top-30 -translate-y-1/2">
            <h2 style={{ fontFamily: 'Roboto Serif', fontWeight: 'regular', fontSize: '60px' }}>Beyond Our Solar System</h2>
            <p style={{ fontFamily: 'Roboto condensed', fontWeight: 'regular', fontSize: '15px', width: '700px'}}>
              ExoGen AI is transforming how we search for planets. 
              With advanced machine learning and real astronomical data, 
              ExoGen identifies and analyzes potential exoplanets, 
              bringing us closer to finding Earth-like worlds and understanding our place in the universe.
            </p>
          </div>
        </span>
        <Image className="relative bottom-55 left-40 z--1" src={bg_planet} alt="EXOGEN AI Logo" width={1000} height={500} priority />
      </div>
      
      <div className="relative bottom-130 right-90">
        <div className="flex flex-col items-center">
          <button className="w-[400px] h-[50px] rounded mt-10 bg-[#F3F0F0] hover:bg-[#e0dcdc] transition-colors duration-150 cursor-pointer">
            <p style={{ fontFamily: 'Roboto condensed', color: "#1B1B1B", fontWeight: 'regular', fontSize: '25px'}}>
              Explore Exoplanets
            </p>
          </button>

          <button className="w-[400px] h-[50px] rounded mt-10 bg-[#F3F0F0] hover:bg-[#e0dcdc] transition-colors duration-150 cursor-pointer">
            <p style={{ fontFamily: 'Roboto condensed', color: "#1B1B1B", fontWeight: 'regular', fontSize: '25px'}}>
              Analyze your Exoplanet
            </p>
          </button>
        </div>
      </div>
    </div>
  );
}
