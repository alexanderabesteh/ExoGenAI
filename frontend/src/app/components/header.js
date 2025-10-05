import React from 'react';
import Image from 'next/image';
import logo from '@/app/assets/svg/EXOGEN_AI_logo.svg';

const Header = () => {
    return (
        <div className="flex justify-center items-center p-4 z-1 relative">
            <span className="absolute left-40" style={{ fontFamily: 'Roboto Serif', fontWeight: 'regular', fontSize: '25px' }}>Explore Resources</span>
            <Image src={logo} alt="EXOGEN AI Logo" width={200} height={200} priority />
            <span className="absolute right-40" style={{ fontFamily: 'Roboto Serif', fontWeight: 'regular', fontSize: '25px' }}>About Us</span>
        </div>
    );
}

export default Header;